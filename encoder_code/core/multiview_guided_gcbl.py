# 本文件用于实现完整模型的边界编码、训练或推理。
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import gcbl_bfanet as gcbl_bfanet


SCHEMA = "boundary-multiview-guided-gcbl-bfanet"
_BASE_BUILD_CASE = edge_gated.mesh_multiclass.build_case


def build_case(*task):
                                                                              
    case = _BASE_BUILD_CASE(*task)
    prediction_path = Path(task[0])
    evidence_root = Path(task[1])
    with np.load(str(evidence_root / (prediction_path.stem + ".npz"))) as handle:
        disagreement = handle["disagreement"].astype(np.float32)
        view_count = handle["view_count"].astype(np.float32)
    if len(disagreement) != len(case["instances"]):
        raise ValueError("B disagreement/mesh length mismatch")
    indices = case["band_indices"]
    coverage = np.clip(view_count / 8.0, 0.0, 1.0)
    cross_view = np.column_stack((
        disagreement[indices],
        coverage[indices],
    )).astype(np.float32)
    case["features"] = np.concatenate(
        (case["features"], cross_view), axis=1
    ).astype(np.float32)
    return case


class MultiViewGuidedBFANetGNN(gcbl_bfanet.GuidedBoundaryBFANetGNN):
                                                                      

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        if input_dim < 3:
            raise ValueError("B requires appended cross-view features")
        super().__init__(input_dim, hidden_dim, layers, dropout)
        self.view_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.view_shift = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.view_shift[-2].weight)
        nn.init.zeros_(self.view_shift[-2].bias)

    def forward(self, x, source, target, edge_u, edge_v):
        disagreement = x[:, -2:-1]
        coverage = x[:, -1:]
        view_state = torch.cat((
            disagreement,
            coverage,
            disagreement * coverage,
        ), dim=1)
        gate = self.view_gate(view_state)
        shift = self.view_shift(view_state)
        modulated = x * (0.75 + 0.5 * gate) + 0.10 * shift
        return super().forward(modulated, source, target, edge_u, edge_v)


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.mesh_multiclass.build_case = build_case
    edge_gated.EdgeGatedGainAwareMeshGNN = MultiViewGuidedBFANetGNN
    edge_gated.calculate_loss = gcbl_bfanet.calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

