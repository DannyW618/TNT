import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, matmul 
from torch_geometric.utils import degree
from typing import Tuple
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    """Graph Convolutional Network (GCN) model."""

    def __init__(self, num_node_features: int, num_hidden: int, num_classes: int,
                 dropout: float = 0.5, use_linear: bool = False, num_layers: int = 2):
        super().__init__()
        self.use_linear = use_linear
        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Number of GCN layers must be at least 1.")

        current_dim = num_node_features
        if use_linear:
            self.lin = torch.nn.Linear(num_node_features, num_hidden)
            current_dim = num_hidden

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(current_dim, num_hidden))

        # Intermediate GCN layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(num_hidden, num_hidden))
        self.output_layer = GCNConv(num_hidden, num_classes)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass using a Data object."""
        x, edge_index = data.x, data.edge_index
        return self._forward_impl(x, edge_index)

    def forward_with_features(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass taking features and edge_index directly."""
        return self._forward_impl(x, edge_index)

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal implementation of the forward pass."""
        if self.use_linear:
            x = self.lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        hidden_representation = x
        logits = self.output_layer(x, edge_index)

        return logits, hidden_representation

    @staticmethod
    def propagation(energy: torch.Tensor, edge_index: torch.Tensor,
                    prop_layers: int = 1, alpha: float = 0.5) -> torch.Tensor:
        """Performs energy belief propagation on the graph."""
        if prop_layers <= 0:
            return energy

        e = energy.unsqueeze(1)
        num_nodes = e.shape[0]
        row, col = edge_index

        node_degree = degree(col, num_nodes, dtype=e.dtype)
        degree_inv = 1.0 / (node_degree[col] + 1e-9)
        norm_value = degree_inv

        adj_matrix = SparseTensor(row=col, col=row, value=norm_value,
                                  sparse_sizes=(num_nodes, num_nodes))

        for _ in range(prop_layers):
            neighbor_energy = matmul(adj_matrix, e)
            e = e * alpha + neighbor_energy * (1 - alpha)

        return e.squeeze(1)

    @torch.no_grad()
    def calculate_energy(self, data, temperature: float = 1.0) -> torch.Tensor:
        """Calculates the negative energy score based on logits."""
        self.eval()
        logits, _ = self(data)
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
        return energy