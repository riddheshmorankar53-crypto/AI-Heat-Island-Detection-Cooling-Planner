"""
Reproducible Training Pipeline for Urban Cooling Planner.

Trains baseline, LightGBM, and Spatial GNN models on the Pune 100m grid dataset.
Saves model artifacts and genuine test set evaluation metrics to `ml/artifacts/metrics.json`.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json

import numpy as np
import torch
import yaml

from data.pipeline import PuneDataPipeline
from ml.features.builder import FEATURE_COLUMNS, SpatialFeaturePipeline
from ml.models.baseline import BaselineHotspotClassifier, BaselineLSTModel
from ml.models.gradient_boost import GradientBoostHotspotClassifier, GradientBoostLSTModel
from ml.models.spatial_gnn import SpatialGNNTrainer, build_spatial_adjacency
from ml.optimizer.calibration import estimate_canopy_cooling


def run_training_pipeline() -> None:
    """Execute end-to-end reproducible training."""
    print("=" * 60)
    print("Starting Pune Urban Cooling Planner Model Training Pipeline")
    print("=" * 60)

    # 1. Load config & data
    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["project"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    artifacts_dir = BASE_DIR / config["paths"]["ml_artifacts"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_pipeline = PuneDataPipeline()
    gdf = data_pipeline.generate_and_save_grid()
    print(f"Loaded {len(gdf)} grid cells for training and evaluation.")

    # 2. Strict train/val/test split before fitting transformers
    feature_pipeline = SpatialFeaturePipeline(seed=seed)
    (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    ) = feature_pipeline.prepare_splits(gdf, test_size=0.15, val_size=0.15)

    print(f"Dataset partitions -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Fit feature scaling on train set only
    X_train_scaled = feature_pipeline.fit_transform(X_train)
    X_test_scaled = feature_pipeline.transform(X_test)

    # Persist scaler artifact
    scaler_path = artifacts_dir / "scaler.joblib"
    feature_pipeline.save_scaler(str(scaler_path))
    print(f"Saved scaler to {scaler_path}")

    metrics_payload = {
        "metadata": {
            "city": "Pune, India",
            "seed": seed,
            "total_cells": len(gdf),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "features_used": FEATURE_COLUMNS,
        },
        "models": {},
    }

    # 3. Train Baseline Models
    print("\n--- Training Baselines ---")
    baseline_reg = BaselineLSTModel(random_state=seed)
    baseline_reg.fit(X_train_scaled, y_reg_train.to_numpy())
    baseline_reg_metrics = baseline_reg.evaluate(X_test_scaled, y_reg_test.to_numpy())
    print(f"Baseline Ridge Regressor Test Metrics: {baseline_reg_metrics}")

    baseline_cls = BaselineHotspotClassifier(random_state=seed)
    baseline_cls.fit(X_train_scaled, y_cls_train.to_numpy())
    baseline_cls_metrics = baseline_cls.evaluate(X_test_scaled, y_cls_test.to_numpy())
    print(f"Baseline Logistic Hotspot Classifier Test Metrics: {baseline_cls_metrics}")

    metrics_payload["models"]["baseline_ridge_lst"] = baseline_reg_metrics
    metrics_payload["models"]["baseline_logistic_hotspot"] = baseline_cls_metrics

    # 4. Train LightGBM LST Regressor
    print("\n--- Training LightGBM LST Regressor ---")
    lgbm_reg = GradientBoostLSTModel(
        n_estimators=config["ml"]["lightgbm"]["n_estimators"],
        learning_rate=config["ml"]["lightgbm"]["learning_rate"],
        max_depth=config["ml"]["lightgbm"]["max_depth"],
        random_state=seed,
    )
    lgbm_reg.fit(X_train, y_reg_train, X_val=X_val, y_val=y_reg_val)
    lgbm_reg_metrics = lgbm_reg.evaluate(X_test, y_reg_test)
    lgbm_reg_fi = lgbm_reg.get_feature_importance(FEATURE_COLUMNS)
    print(f"LightGBM LST Regressor Test Metrics: {lgbm_reg_metrics}")
    print(
        f"Top Features for LST: {sorted(lgbm_reg_fi.items(), key=lambda x: x[1], reverse=True)[:4]}"
    )

    lgbm_reg_path = artifacts_dir / "lgbm_lst_model.joblib"
    lgbm_reg.save(str(lgbm_reg_path))
    metrics_payload["models"]["lightgbm_lst_regressor"] = {
        **lgbm_reg_metrics,
        "feature_importances": lgbm_reg_fi,
    }

    # 5. Train LightGBM Hotspot Classifier
    print("\n--- Training LightGBM Hotspot Classifier ---")
    lgbm_cls = GradientBoostHotspotClassifier(
        n_estimators=config["ml"]["lightgbm"]["n_estimators"],
        learning_rate=config["ml"]["lightgbm"]["learning_rate"],
        max_depth=config["ml"]["lightgbm"]["max_depth"],
        random_state=seed,
    )
    lgbm_cls.fit(X_train, y_cls_train, X_val=X_val, y_val=y_cls_val)
    lgbm_cls_metrics = lgbm_cls.evaluate(X_test, y_cls_test)
    lgbm_cls_fi = lgbm_cls.get_feature_importance(FEATURE_COLUMNS)
    print(f"LightGBM Hotspot Classifier Test Metrics: {lgbm_cls_metrics}")

    lgbm_cls_path = artifacts_dir / "lgbm_hotspot_model.joblib"
    lgbm_cls.save(str(lgbm_cls_path))
    metrics_payload["models"]["lightgbm_hotspot_classifier"] = {
        **lgbm_cls_metrics,
        "feature_importances": lgbm_cls_fi,
    }

    # 6. Train PyTorch Spatial GNN
    print("\n--- Training PyTorch Spatial Microclimate GNN ---")
    # Build spatial graph adjacency over entire grid
    adj_norm = build_spatial_adjacency(gdf)

    # Full tensor features & targets
    full_X = gdf[FEATURE_COLUMNS].to_numpy()
    scaler_full = feature_pipeline.scaler
    full_X_scaled = scaler_full.transform(full_X)

    features_tensor = torch.tensor(full_X_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(gdf["lst_celsius"].to_numpy(), dtype=torch.float32)

    # Masks corresponding to train/val/test indices
    n_total = len(gdf)
    train_indices = X_train.index.to_numpy()
    val_indices = X_val.index.to_numpy()
    test_indices = X_test.index.to_numpy()

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    gnn_trainer = SpatialGNNTrainer(
        in_features=len(FEATURE_COLUMNS),
        hidden_dim=config["ml"]["gnn"]["hidden_dim"],
        lr=config["ml"]["gnn"]["learning_rate"],
        seed=seed,
    )
    gnn_train_res = gnn_trainer.train(
        features=features_tensor,
        adj_norm=adj_norm,
        targets=targets_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=config["ml"]["gnn"]["epochs"],
    )
    gnn_test_metrics = gnn_trainer.evaluate(
        features=features_tensor,
        adj_norm=adj_norm,
        targets=targets_tensor,
        test_mask=test_mask,
    )
    print(f"Spatial GNN Test Metrics: {gnn_test_metrics}")

    gnn_path = artifacts_dir / "spatial_gnn.pt"
    gnn_trainer.save(str(gnn_path))
    metrics_payload["models"]["spatial_microclimate_gnn"] = {
        **gnn_test_metrics,
        **gnn_train_res,
    }

    # 7. Measured intervention effects (used by the optimizer's tree catalog entry)
    print("\n--- Measuring tree canopy cooling effect ---")
    canopy_effect = estimate_canopy_cooling(gdf, seed=seed)
    print(f"Canopy effect: {canopy_effect}")
    metrics_payload["intervention_effects"] = {"tree_canopy": canopy_effect}

    # 8. Persist real metrics JSON
    metrics_file = artifacts_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved real evaluation metrics to {metrics_file}")
    print("=" * 60)
    print("Model Training Pipeline Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
