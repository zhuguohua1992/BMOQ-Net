# 本文件用于实现完整模型的边界编码、训练或推理。
import torch
import torch.nn as nn

import edge_gated_gnn as edge_gated
import mesh_attention_gnn as mesh_attention
import boundary_tversky_attention as boundary_tversky


class MultiScaleBoundarySemanticGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        layers = max(int(layers), 6)
        self.encoder = edge_gated.mesh_multiclass.MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            mesh_attention.MeshAttentionBlock(hidden_dim, 4, dropout) for _ in range(layers)
        ])
        self.scale_fusion = edge_gated.mesh_multiclass.MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.boundary_semantic = edge_gated.mesh_multiclass.MLP(
            hidden_dim * 3, hidden_dim, hidden_dim, dropout
        )
        self.boundary_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.context = edge_gated.mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 17)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 17),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, source, target, edge_u, edge_v):
        hidden = self.encoder(x)
        snapshots = []
        sample_layers = {0, 2, len(self.blocks) - 1}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, source, target)
            if index in sample_layers:
                snapshots.append(hidden)
        multi_scale = torch.cat(snapshots, dim=1)
        base = self.scale_fusion(multi_scale)
        boundary_residual = self.boundary_semantic(multi_scale)
        gate = self.boundary_gate(multi_scale)
        hidden = self.fusion_norm(base + gate * boundary_residual)
        global_context = self.context(torch.cat((
            hidden.mean(dim=0, keepdim=True),
            hidden.max(dim=0, keepdim=True)[0],
        ), dim=1))
        hidden = self.context_norm(hidden + global_context)
        edge_pair = torch.cat((
            torch.abs(hidden[edge_u] - hidden[edge_v]),
            hidden[edge_u] * hidden[edge_v],
        ), dim=1)
        return (
            self.class_head(hidden),
            self.boundary_head(hidden).squeeze(1),
            torch.tanh(self.gain_head(hidden)),
            self.edge_head(edge_pair).squeeze(1),
        )


def main():
    edge_gated.SCHEMA = "boundary-multiscale-boundary-semantic-mesh-attention"
    edge_gated.EdgeGatedGainAwareMeshGNN = MultiScaleBoundarySemanticGNN
    edge_gated.calculate_loss = boundary_tversky.calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

