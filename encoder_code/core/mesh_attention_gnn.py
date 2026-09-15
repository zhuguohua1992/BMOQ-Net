# 本文件用于实现完整模型的边界编码、训练或推理。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import edge_gated_gnn as edge_gated


class MeshAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = hidden_dim // self.heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.heads),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, source, target):
        node_count = hidden.shape[0]
        q = self.query(hidden).view(node_count, self.heads, self.head_dim)
        k = self.key(hidden).view(node_count, self.heads, self.head_dim)
        v = self.value(hidden).view(node_count, self.heads, self.head_dim)
        logits = (q[source] * k[target]).sum(dim=2) / math.sqrt(float(self.head_dim))
        pair = torch.cat((torch.abs(hidden[source] - hidden[target]), hidden[source] * hidden[target]), dim=1)
        gate = torch.sigmoid(self.edge_gate(pair))
        weight = torch.exp(torch.clamp(logits, -8.0, 8.0)) * (0.05 + gate)
        denominator = hidden.new_zeros((node_count, self.heads))
        denominator.index_add_(0, source, weight)
        aggregate = hidden.new_zeros((node_count, self.heads, self.head_dim))
        aggregate.index_add_(0, source, weight[:, :, None] * v[target])
        aggregate = aggregate / denominator.clamp_min(1.0e-6)[:, :, None]
        message = self.output(aggregate.reshape(node_count, -1))
        hidden = self.norm1(hidden + self.dropout(message))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class MeshAttentionGainGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.encoder = edge_gated.mesh_multiclass.MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            MeshAttentionBlock(hidden_dim, 4, dropout) for _ in range(int(layers))
        ])
        self.context = edge_gated.mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 17)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 17)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, source, target, edge_u, edge_v):
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden, source, target)
        context = self.context(torch.cat((hidden.mean(dim=0, keepdim=True), hidden.max(dim=0, keepdim=True)[0]), dim=1))
        hidden = self.norm(hidden + context)
        edge_pair = torch.cat((torch.abs(hidden[edge_u] - hidden[edge_v]), hidden[edge_u] * hidden[edge_v]), dim=1)
        return (
            self.class_head(hidden),
            self.boundary_head(hidden).squeeze(1),
            torch.tanh(self.gain_head(hidden)),
            self.edge_head(edge_pair).squeeze(1),
        )


def main():
    edge_gated.SCHEMA = "boundary-mesh-attention-gain-gnn"
    edge_gated.EdgeGatedGainAwareMeshGNN = MeshAttentionGainGNN
    edge_gated.main()


if __name__ == "__main__":
    main()

