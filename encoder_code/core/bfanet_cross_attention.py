# 本文件用于实现完整模型的边界编码、训练或推理。
import math

import torch
import torch.nn as nn

import edge_gated_gnn as edge_gated
import mesh_attention_gnn as mesh_attention
import boundary_tversky_attention as boundary_tversky
import soft_semantic_boundary_iou as boundary_soft_semantic


SCHEMA = "boundary-bfanet-boundary-semantic-cross-attention"


class BFANetMeshCrossAttentionGNN(nn.Module):
                                                                              

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        layers = max(int(layers), 6)
        heads = 4
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by four attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.encoder = edge_gated.mesh_multiclass.MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            mesh_attention.MeshAttentionBlock(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])

        multi_dim = hidden_dim * 3
        self.semantic_projection = edge_gated.mesh_multiclass.MLP(
            multi_dim, hidden_dim, hidden_dim, dropout
        )
        self.boundary_projection = edge_gated.mesh_multiclass.MLP(
            multi_dim, hidden_dim, hidden_dim, dropout
        )
        self.semantic_query = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_query = nn.Linear(hidden_dim, hidden_dim)
        self.fused_query = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_key = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_value = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_key = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_value = nn.Linear(hidden_dim, hidden_dim)

        pair_dim = hidden_dim * 4
        self.edge_gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, heads * 2),
        )
        self.semantic_message = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_message = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_injection_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.semantic_norm = nn.LayerNorm(hidden_dim)
        self.boundary_norm = nn.LayerNorm(hidden_dim)
        self.semantic_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.semantic_ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.context = edge_gated.mesh_multiclass.MLP(
            hidden_dim * 2, hidden_dim, hidden_dim, dropout
        )
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

    def _neighbour_aggregate(self, query, key, value, gate, source, target):
        node_count = query.shape[0]
        query = query.view(node_count, self.heads, self.head_dim)
        key = key.view(node_count, self.heads, self.head_dim)
        value = value.view(node_count, self.heads, self.head_dim)
        logits = (
            query[source] * key[target]
        ).sum(dim=2) / math.sqrt(float(self.head_dim))
        weight = torch.exp(torch.clamp(logits, -8.0, 8.0)) * (
            0.05 + torch.sigmoid(gate)
        )
        denominator = query.new_zeros((node_count, self.heads))
        denominator.index_add_(0, source, weight)
        aggregate = query.new_zeros((node_count, self.heads, self.head_dim))
        aggregate.index_add_(0, source, weight[:, :, None] * value[target])
        aggregate = aggregate / denominator.clamp_min(1.0e-6)[:, :, None]
        return aggregate.reshape(node_count, -1)

    def forward(self, x, source, target, edge_u, edge_v):
        hidden = self.encoder(x)
        snapshots = []
        sample_layers = {0, 2, len(self.blocks) - 1}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, source, target)
            if index in sample_layers:
                snapshots.append(hidden)
        multi_scale = torch.cat(snapshots, dim=1)

        semantic = self.semantic_projection(multi_scale)
        boundary = self.boundary_projection(multi_scale)
        fused_query = self.fused_query(torch.cat((
            self.semantic_query(semantic),
            self.boundary_query(boundary),
        ), dim=1))
        pair = torch.cat((
            torch.abs(semantic[source] - semantic[target]),
            semantic[source] * semantic[target],
            torch.abs(boundary[source] - boundary[target]),
            boundary[source] * boundary[target],
        ), dim=1)
        semantic_gate, boundary_gate = self.edge_gate(pair).chunk(2, dim=1)
        semantic_context = self._neighbour_aggregate(
            fused_query,
            self.semantic_key(semantic),
            self.semantic_value(semantic),
            semantic_gate,
            source,
            target,
        )
        boundary_context = self._neighbour_aggregate(
            fused_query,
            self.boundary_key(boundary),
            self.boundary_value(boundary),
            boundary_gate,
            source,
            target,
        )
        boundary_hidden = self.boundary_norm(
            boundary + self.dropout(self.boundary_message(boundary_context))
        )
        injection_gate = self.boundary_injection_gate(
            torch.cat((semantic, boundary_hidden), dim=1)
        )
        semantic_hidden = self.semantic_norm(
            semantic
            + self.dropout(self.semantic_message(semantic_context))
            + injection_gate * self.dropout(self.boundary_message(boundary_context))
        )
        semantic_hidden = self.semantic_ffn_norm(
            semantic_hidden + self.dropout(self.semantic_ffn(semantic_hidden))
        )

        global_context = self.context(torch.cat((
            semantic_hidden.mean(dim=0, keepdim=True),
            semantic_hidden.max(dim=0, keepdim=True)[0],
        ), dim=1))
        semantic_hidden = self.context_norm(semantic_hidden + global_context)
        edge_pair = torch.cat((
            torch.abs(semantic_hidden[edge_u] - semantic_hidden[edge_v]),
            semantic_hidden[edge_u] * semantic_hidden[edge_v],
        ), dim=1)
        return (
            self.class_head(semantic_hidden),
            self.boundary_head(boundary_hidden).squeeze(1),
            torch.tanh(self.gain_head(semantic_hidden)),
            self.edge_head(edge_pair).squeeze(1),
        )


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.EdgeGatedGainAwareMeshGNN = BFANetMeshCrossAttentionGNN
    edge_gated.calculate_loss = boundary_soft_semantic.calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

