# Mathematical Formulations: Urban Cooling Planner

This document provides the mathematical specifications for the Heat Vulnerability Index, the Google OR-Tools Mixed-Integer Linear Program (MILP) cooling intervention optimizer, and the pedestrian heat-safe walking route cost function.

---

## 1. Heat Vulnerability Index (HVI)

The composite Heat Vulnerability Index ($\text{HVI}_i \in [0, 1]$) for a 100m grid cell $i$ is formulated following IPCC climate risk frameworks combining **Exposure**, **Sensitivity**, and **Adaptive Capacity**:

$$\text{HVI}_i = w_E \cdot E_i + w_S \cdot S_i + w_A \cdot (1 - A_i)$$

Where:
- **Weights**: $w_E = 0.40$ (Thermal Exposure), $w_S = 0.35$ (Socioeconomic Sensitivity), $w_A = 0.25$ (Adaptive Capacity).
- **Exposure ($E_i$)**: Min-max normalized summer daytime Land Surface Temperature:
  $$E_i = \frac{\text{LST}_i - \min(\text{LST})}{\max(\text{LST}) - \min(\text{LST})}$$
- **Sensitivity ($S_i$)**: Population density and vulnerable demographics (elderly $>65$, children $<5$, low-income informal settlements):
  $$S_i = 0.60 \cdot \text{VulnRatio}_i + 0.40 \cdot \frac{\text{PopDensity}_i}{\max(\text{PopDensity})}$$
- **Adaptive Capacity ($A_i$)**: Protective natural and built buffers:
  $$A_i = 0.50 \cdot \left(\frac{\text{Canopy}_i}{100}\right) + 0.50 \cdot \left(1 - \min\left(1.0, \frac{\text{DistCoolingFacility}_i}{2000}\right)\right)$$

Cells with $\text{LST}_i \ge P_{82}(\text{LST})$ and $\text{HVI}_i \ge 0.62$ are classified as **Critical Hotspots**.

---

## 2. Cooling Intervention Optimization (OR-Tools MILP)

### Decision Variables
Let $i \in \{1, \dots, N\}$ index candidate 100m grid cells, and $k \in \mathcal{K} = \{\text{trees}, \text{cool\_roofs}, \text{water\_points}, \text{shaded\_stops}, \text{cooling\_centers}\}$ index intervention categories.

$$x_{i, k} \in \{0, 1\} \quad \forall i, k$$

where $x_{i, k} = 1$ if intervention $k$ is deployed at cell $i$, and $0$ otherwise.

### Constraints

1. **Total Municipal Budget Constraint**:
   $$\sum_{i=1}^N \sum_{k \in \mathcal{K}} c_k \cdot x_{i, k} \le B$$
   where $c_k$ is the unit capital cost of intervention $k$ (in INR), and $B$ is the allocated budget.

2. **Per-Cell Capacity Constraint**:
   $$\sum_{k \in \mathcal{K}} x_{i, k} \le M_{\text{cell}} \quad \forall i$$
   (typically $M_{\text{cell}} = 1$ to prevent spatial overcrowding).

3. **Intervention Suitability Constraints**:
   $$x_{i, k} = 0 \quad \text{if } \neg \text{Suitable}(i, k)$$
   - Trees: $\text{Canopy}_i \le 35\%$
   - Cool Roofs: $\text{Impervious}_i \ge 45\%$
   - Water Points: $\text{HVI}_i \ge 0.45$
   - Shaded Bus Stops: $\text{IsTransitHub}_i = 1 \lor \text{PopDensity}_i \ge 20,000$
   - Cooling Centers: $\text{HVI}_i \ge 0.65$

4. **Cooling Center Minimum Spatial Separation**:
   To maximize geographical equity and prevent redundancy, no two community cooling shelters may be placed within $400\text{ meters}$ of each other:
   $$x_{i, \text{cc}} + x_{j, \text{cc}} \le 1 \quad \forall (i, j) \text{ such that } d(i, j) < 400\text{ m}$$

5. **Portfolio Diversity Constraint**:
   When $\ge 3$ intervention types are requested under a municipal budget $B \ge \text{₹}300,000$, no single intervention type may absorb more than $45\%$ of the total budget:
   $$\sum_{i=1}^N c_k \cdot x_{i, k} \le 0.45 \cdot B \quad \forall k \in \mathcal{K}$$

### Objective Function
Maximize the combined heat reduction, demographic vulnerability relief, and spatial population coverage:

$$\max \sum_{i=1}^N \sum_{k \in \mathcal{K}} \left[ \alpha \cdot \left( \Delta T_k \cdot \text{HVI}_i \cdot \left(\frac{R_k}{50}\right)^{1.5} \right) + \beta \cdot \left( \text{Pop}_i \cdot \frac{R_k}{50} \right) \right] x_{i, k}$$

where $\Delta T_k$ is the empirical cooling drop (°C), $R_k$ is the physical radius of influence (meters), $\text{HVI}_i$ is cell heat vulnerability, and $\text{Pop}_i$ is the resident population.

---

## 3. Heat-Safe Walking Routing Engine

Let $G = (V, E)$ represent the pedestrian street network of Pune, where each edge $e = (u, v) \in E$ is characterized by:
- $L_e$: Physical segment length in meters.
- $S_e \in [0, 1]$: Tree canopy shade fraction covering the street.
- $H_e \in [0, 1]$: Normalized thermal risk index of the intersecting 100m grid cell:
  $$H_e = \text{clip}\left( \frac{\text{LST}_e - 30.0}{14.0}, 0.05, 1.0 \right)$$

### Dual-Objective Formulation

1. **Standard Fastest Route**:
   $$P_{\text{fast}} = \arg\min_{P \in \mathcal{P}(s, t)} \sum_{e \in P} L_e$$

2. **Heat-Safe Cool Route**:
   $$P_{\text{cool}} = \arg\min_{P \in \mathcal{P}(s, t)} \sum_{e \in P} L_e \cdot \left( 1 + \gamma \cdot H_e \cdot (1 - S_e) \right)$$
   where $\gamma \ge 0$ is the citizen's heat avoidance sensitivity parameter.

### Thermal Stress Reduction Metric
The relative reduction in pedestrian heat stress achieved by the heat-safe route compared to the direct route:

$$\Delta \text{Stress} = \left( 1 - \frac{\sum_{e \in P_{\text{cool}}} L_e \cdot H_e \cdot (1 - 0.75 S_e)}{\sum_{e \in P_{\text{fast}}} L_e \cdot H_e \cdot (1 - 0.75 S_e)} \right) \times 100\%$$
