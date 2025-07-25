import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import config

class GaussianFilter(nn.Module):
    def __init__(self, start=0.0, stop=config.CUTOFF_RADIUS, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.gamma = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.gamma * torch.pow(dist, 2))

class InteractionBlock(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.node_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        aggregated_messages = self.propagate(edge_index, x=x)
        x = x + self.node_update_mlp(torch.cat([x, aggregated_messages], dim=-1))
        x = self.norm(x)
        row, col = edge_index
        edge_attr = edge_attr + self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        return x, edge_attr

    def message(self, x_j):
        return x_j

class GNNForceField(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, num_layers):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim, hidden_dim)
        self.edge_embedding = GaussianFilter(num_gaussians=edge_dim)
        self.interaction_blocks = nn.ModuleList(
            [InteractionBlock(hidden_dim, edge_dim) for _ in range(num_layers)]
        )
        self.force_readout = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for block in self.interaction_blocks:
            x, edge_attr = block(x, edge_index, edge_attr)
        force_magnitudes = self.force_readout(edge_attr).squeeze(-1)
        row, col = edge_index
        unit_vectors = (pos[row] - pos[col]) / (torch.norm(pos[row] - pos[col], dim=1, keepdim=True) + 1e-8)
        force_contributions = force_magnitudes.unsqueeze(1) * unit_vectors
        final_forces = torch.zeros_like(pos)
        final_forces.index_add_(0, col, force_contributions)
        return final_forces