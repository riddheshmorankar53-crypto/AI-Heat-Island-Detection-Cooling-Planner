"""
Spatial Graph Convolutional Network (GCN) for microclimate heat modeling.

Constructs spatial adjacency across 100m grid cells and applies 2-layer graph
convolution to capture thermal diffusion and neighborhood microclimate effects.
Implemented in PyTorch for robust, reproducible execution.
"""

from typing import Any, Dict

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class GraphConvLayer(nn.Module):
    """Spatial Graph Convolution: H^(l+1) = D^(-1/2) * A * D^(-1/2) * H^(l) * W."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # Spatial neighborhood aggregation
        support = self.linear(x)
        output = torch.matmul(adj_norm, support)
        return output


class SpatialMicroclimateGNN(nn.Module):
    """2-Layer Spatial GNN for Land Surface Temperature & Heat Vulnerability estimation."""

    def __init__(
        self, in_features: int, hidden_dim: int = 32, out_features: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.conv1 = GraphConvLayer(in_features, hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GraphConvLayer(hidden_dim, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj_norm)
        h = self.act1(h)
        h = self.dropout(h)
        out = self.conv2(h, adj_norm)
        return out


def build_spatial_adjacency(
    gdf: gpd.GeoDataFrame, distance_threshold_deg: float = 0.0018
) -> torch.Tensor:
    """
    Construct normalized spatial adjacency matrix from 100m grid cell centroids.
    Two cells are adjacent if their Euclidean distance <= ~150 meters (0.0018 degrees).

    Returns:
        Symmetrically normalized adjacency tensor (N, N).
    """
    coords = np.column_stack([gdf["centroid_lon"].to_numpy(), gdf["centroid_lat"].to_numpy()])
    # Compute pairwise Euclidean distances efficiently
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # Binary adjacency + self-loops
    adj = (dist <= distance_threshold_deg).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    # Symmetric degree normalization: D^(-0.5) * A * D^(-0.5)
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid_mask = degree > 0
    degree_inv_sqrt[valid_mask] = 1.0 / np.sqrt(degree[valid_mask])
    d_mat = np.diag(degree_inv_sqrt)

    norm_adj = d_mat @ adj @ d_mat
    return torch.tensor(norm_adj, dtype=torch.float32)


class SpatialGNNTrainer:
    """Trains and evaluates the Spatial Microclimate GNN."""

    def __init__(
        self, in_features: int, hidden_dim: int = 32, lr: float = 0.01, seed: int = 42
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = SpatialMicroclimateGNN(
            in_features=in_features, hidden_dim=hidden_dim, out_features=1
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.target_mean = 0.0
        self.target_std = 1.0

    def train(
        self,
        features: torch.Tensor,
        adj_norm: torch.Tensor,
        targets: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        epochs: int = 150,
    ) -> Dict[str, Any]:
        # Normalize targets based on training split
        train_targets = targets[train_mask]
        self.target_mean = float(torch.mean(train_targets).item())
        self.target_std = float(torch.std(train_targets).item())
        if self.target_std == 0:
            self.target_std = 1.0

        scaled_targets = (targets - self.target_mean) / self.target_std

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(features, adj_norm).squeeze()
            loss = self.criterion(out[train_mask], scaled_targets[train_mask])
            loss.backward()
            self.optimizer.step()

        # Final losses in original Celsius scale
        self.model.eval()
        with torch.no_grad():
            preds_norm = self.model(features, adj_norm).squeeze()
            preds = preds_norm * self.target_std + self.target_mean
            train_rmse = float(
                torch.sqrt(torch.mean((preds[train_mask] - targets[train_mask]) ** 2)).item()
            )
            val_rmse = float(
                torch.sqrt(torch.mean((preds[val_mask] - targets[val_mask]) ** 2)).item()
            )

        return {"train_rmse": round(train_rmse, 4), "val_rmse": round(val_rmse, 4)}

    def evaluate(
        self,
        features: torch.Tensor,
        adj_norm: torch.Tensor,
        targets: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            preds_norm = self.model(features, adj_norm).squeeze()[test_mask]
            preds = (preds_norm * self.target_std + self.target_mean).cpu().numpy()
            y_true = targets[test_mask].cpu().numpy()

        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        mae = float(np.mean(np.abs(y_true - preds)))
        ss_res = np.sum((y_true - preds) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        self.model.eval()
