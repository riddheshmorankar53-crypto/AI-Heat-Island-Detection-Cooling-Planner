"""
Google OR-Tools Mixed-Integer Linear Programming (MILP) solver for
urban cooling intervention planning under municipal budget constraints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from ortools.linear_solver import pywraplp

METRICS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "metrics.json"
# Calibration knob: canopy one ₹15,000 tree unit adds across its 50m influence area at
# maturity (~25 street trees ≈ 750 m² crown ≈ 10% of 7,850 m²). Tune to local costs.
TREE_CANOPY_GAIN_PCT_PER_UNIT = 10.0
TREE_LITERATURE_DELTA_C = 1.5
LITERATURE_BASIS = "Literature assumption (not measurable from satellite LST)"


def _tree_cooling() -> Tuple[float, str]:
    """Tree cooling per unit from the canopy effect measured on the Pune grid by ml/train.py."""
    try:
        effect = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["intervention_effects"][
            "tree_canopy"
        ]
    except (OSError, KeyError, ValueError):
        return TREE_LITERATURE_DELTA_C, "Literature assumption (run ml/train.py to measure)"
    per_pct = -effect["lst_change_per_canopy_pct"]
    lo, hi = sorted(-c * TREE_CANOPY_GAIN_PCT_PER_UNIT for c in effect["ci95"])
    basis = (
        f"Measured in Pune: {per_pct:.3f}°C per +1% canopy x {TREE_CANOPY_GAIN_PCT_PER_UNIT:g}% "
        f"per unit (95% CI {lo:.2f}-{hi:.2f}°C)"
    )
    return round(per_pct * TREE_CANOPY_GAIN_PCT_PER_UNIT, 2), basis


_TREE_DELTA, _TREE_BASIS = _tree_cooling()

INTERVENTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "trees": {
        "name": "Urban Tree Canopy",
        "cost_inr": 15000.0,
        "delta_celsius": _TREE_DELTA,
        "basis": _TREE_BASIS,
        "influence_radius_m": 50.0,
        "suitability": lambda row: row["tree_canopy_pct"] <= 35.0,
        "color": "#10b981",  # emerald
        "icon": "trees",
    },
    "cool_roofs": {
        "name": "High-Albedo Cool Roofs",
        "cost_inr": 35000.0,
        "delta_celsius": 2.2,
        "basis": "Literature assumption (cross-section confounded: bright bare soil is hot)",
        "influence_radius_m": 30.0,
        "suitability": lambda row: row["impervious_pct"] >= 45.0,
        "color": "#06b6d4",  # cyan
        "icon": "roof",
    },
    "water_points": {
        "name": "Misting & Hydration Station",
        "cost_inr": 50000.0,
        "delta_celsius": 1.0,
        "basis": LITERATURE_BASIS,
        "influence_radius_m": 40.0,
        "suitability": lambda row: row["heat_vulnerability_index"] >= 0.45,
        "color": "#3b82f6",  # blue
        "icon": "droplets",
    },
    "shaded_stops": {
        "name": "Shaded Transit Stops",
        "cost_inr": 80000.0,
        "delta_celsius": 2.0,
        "basis": LITERATURE_BASIS,
        "influence_radius_m": 60.0,
        "suitability": lambda row: (
            bool(row["is_transit_hub"]) or (row["pop_density_sqkm"] >= 20000)
        ),
        "color": "#f59e0b",  # amber
        "icon": "umbrella",
    },
    "cooling_centers": {
        "name": "Community Cooling Refuge",
        "cost_inr": 250000.0,
        "delta_celsius": 4.5,
        "basis": LITERATURE_BASIS,
        "influence_radius_m": 350.0,
        "suitability": lambda row: row["heat_vulnerability_index"] >= 0.65,
        "color": "#ec4899",  # pink
        "icon": "shield-check",
    },
}


class CoolingInterventionOptimizer:
    """Solves optimal spatial placement of cooling interventions within budget constraints."""

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.catalog = catalog or INTERVENTION_CATALOG

    def optimize(
        self,
        grid_gdf: gpd.GeoDataFrame,
        budget_inr: float = 1500000.0,
        enabled_types: Optional[List[str]] = None,
        max_interventions_per_cell: int = 1,
        priority_hotspots_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve MILP for cooling intervention placement.

        Args:
            grid_gdf: GeoDataFrame containing 100m cells and attributes.
            budget_inr: Total municipal budget in Indian Rupees (INR).
            enabled_types: List of allowed intervention keys (e.g. ['trees', 'cool_roofs']).
            max_interventions_per_cell: Maximum interventions allowed per 100m cell.
            priority_hotspots_only: If True, restricts candidates to critical hotspot cells.

        Returns:
            Dictionary containing GeoJSON feature collection of chosen interventions,
            total cost, temperature reduction, population covered, and budget utilization.
        """
        if enabled_types is None:
            enabled_types = list(self.catalog.keys())
        else:
            enabled_types = [t for t in enabled_types if t in self.catalog]

        if not enabled_types:
            raise ValueError("At least one valid intervention type must be enabled.")

        # Filter candidate cells
        candidates = (
            grid_gdf[grid_gdf["is_critical_hotspot"]] if priority_hotspots_only else grid_gdf
        )

        # Limit to top candidate cells if grid is large for snappy laptop performance (<1.5 sec)
        # Sort primarily by heat vulnerability index and LST
        if len(candidates) > 800:
            candidates = candidates.sort_values(
                by=["heat_vulnerability_index", "lst_celsius"], ascending=[False, False]
            ).head(800)

        candidates = candidates.reset_index(drop=True)
        num_cells = len(candidates)

        # Create SCIP solver via OR-Tools
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            # Fallback to CBC
            solver = pywraplp.Solver.CreateSolver("CBC")
            if not solver:
                raise RuntimeError("No suitable MILP solver (SCIP/CBC) available in OR-Tools.")

        # Decision variables: x[i, k] in {0, 1}
        x: Dict[Tuple[int, str], pywraplp.Variable] = {}
        for i in range(num_cells):
            row = candidates.iloc[i]
            for k in enabled_types:
                interv = self.catalog[k]
                # Check suitability
                is_suitable = interv["suitability"](row)
                if is_suitable:
                    x[i, k] = solver.BoolVar(f"x_{i}_{k}")

        # 1. Budget constraint: sum(cost[k] * x[i, k]) <= budget
        budget_constraint = solver.Constraint(0.0, float(budget_inr), "BudgetConstraint")
        for (i, k), var in x.items():
            cost = self.catalog[k]["cost_inr"]
            budget_constraint.SetCoefficient(var, cost)

        # 2. Maximum interventions per cell constraint
        for i in range(num_cells):
            cell_vars = [x[i, k] for k in enabled_types if (i, k) in x]
            if cell_vars:
                cell_constraint = solver.Constraint(
                    0.0, float(max_interventions_per_cell), f"CellCap_{i}"
                )
                for var in cell_vars:
                    cell_constraint.SetCoefficient(var, 1.0)

        # 3. Cooling Center minimum distance separation constraint (no two within 400m)
        if "cooling_centers" in enabled_types:
            cc_indices = [i for i in range(num_cells) if (i, "cooling_centers") in x]
            coords = candidates.loc[cc_indices, ["centroid_lon", "centroid_lat"]].to_numpy()
            n_cc = len(cc_indices)
            for idx_a in range(n_cc):
                for idx_b in range(idx_a + 1, n_cc):
                    d_lon = (coords[idx_a, 0] - coords[idx_b, 0]) * 105000.0
                    d_lat = (coords[idx_a, 1] - coords[idx_b, 1]) * 111000.0
                    if np.hypot(d_lon, d_lat) < 400.0:
                        i_a = cc_indices[idx_a]
                        i_b = cc_indices[idx_b]
                        sep_constraint = solver.Constraint(0.0, 1.0, f"CC_Sep_{i_a}_{i_b}")
                        sep_constraint.SetCoefficient(x[i_a, "cooling_centers"], 1.0)
                        sep_constraint.SetCoefficient(x[i_b, "cooling_centers"], 1.0)

        # 4. Portfolio diversity constraint: if multiple types enabled, prevent 100% budget monopoly
        if len(enabled_types) >= 3 and budget_inr >= 300000.0:
            max_share = 0.45 * float(budget_inr)
            for k in enabled_types:
                type_vars = [x[i, k] for i in range(num_cells) if (i, k) in x]
                if type_vars:
                    cost_k = self.catalog[k]["cost_inr"]
                    type_budget_constraint = solver.Constraint(0.0, max_share, f"TypeMaxBudget_{k}")
                    for var in type_vars:
                        type_budget_constraint.SetCoefficient(var, cost_k)

        # 5. Objective: Maximize cooling impact * vulnerability * population relief
        objective = solver.Objective()
        objective.SetMaximization()
        for (i, k), var in x.items():
            row = candidates.iloc[i]
            interv = self.catalog[k]
            delta_t = interv["delta_celsius"]
            vuln = float(row["heat_vulnerability_index"])
            pop = float(row["population"])
            radius = interv["influence_radius_m"]

            # Grounded benefit: cooling intensity * vulnerability * spatial radius + direct pop coverage
            area_weight = (radius / 50.0) ** 1.5
            benefit_score = 500.0 * (delta_t * vuln * area_weight) + 0.5 * (pop * (radius / 50.0))
            objective.SetCoefficient(var, float(benefit_score))

        # Solve
        solver.set_time_limit(10000)  # max 10s
        status = solver.Solve()

        # Extract recommendations
        selected_features = []
        total_cost = 0.0
        total_pop_benefited = 0
        type_counts: Dict[str, int] = {k: 0 for k in enabled_types}
        total_cooling_points = 0.0

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            for (i, k), var in x.items():
                if var.solution_value() > 0.5:
                    row = candidates.iloc[i]
                    interv = self.catalog[k]
                    cost = interv["cost_inr"]
                    total_cost += cost
                    type_counts[k] += 1
                    total_pop_benefited += int(row["population"])
                    total_cooling_points += interv["delta_celsius"]

                    selected_features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    float(row["centroid_lon"]),
                                    float(row["centroid_lat"]),
                                ],
                            },
                            "properties": {
                                "cell_id": str(row["cell_id"]),
                                "intervention_type": k,
                                "intervention_name": interv["name"],
                                "cost_inr": cost,
                                "cooling_delta_celsius": interv["delta_celsius"],
                                "influence_radius_m": interv["influence_radius_m"],
                                "color": interv["color"],
                                "icon": interv["icon"],
                                "current_lst": float(row["lst_celsius"]),
                                "hvi": float(row["heat_vulnerability_index"]),
                                "cell_population": int(row["population"]),
                            },
                        }
                    )

        avg_cooling_delta = round(total_cooling_points / max(1, len(selected_features)), 2)
        budget_utilization_pct = round((total_cost / max(1.0, budget_inr)) * 100.0, 1)

        return {
            "status": "OPTIMAL"
            if status == pywraplp.Solver.OPTIMAL
            else "FEASIBLE"
            if status == pywraplp.Solver.FEASIBLE
            else "INFEASIBLE",
            "summary": {
                "budget_allocated_inr": budget_inr,
                "total_cost_inr": total_cost,
                "budget_utilization_pct": budget_utilization_pct,
                "interventions_count": len(selected_features),
                "type_breakdown": type_counts,
                "avg_cooling_impact_celsius": avg_cooling_delta,
                "total_population_protected": total_pop_benefited,
            },
            "features": selected_features,
        }
