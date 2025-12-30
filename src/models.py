"""
Graph Neural Network models for mesh super-resolution.
Maps coarse mesh CFD simulation results to fine mesh resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple
import numpy as np


class MeshGNN(nn.Module):
    """
    Graph Neural Network for mesh-based field super-resolution.
    Uses message passing to propagate information between mesh nodes.
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 conv_type: str = 'GAT'):
        """
        Initialize the mesh GNN.
        
        Args:
            in_channels: Number of input features (e.g., velocity_x, velocity_y, pressure, temperature + coordinates)
            hidden_channels: Hidden dimension size
            out_channels: Number of output features
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of convolution ('GCN', 'GAT', 'SAGE')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == 'GCN':
                conv = GCNConv(hidden_channels, hidden_channels)
            elif conv_type == 'GAT':
                conv = GATConv(hidden_channels, hidden_channels // 8, heads=8, concat=True, dropout=dropout)
            elif conv_type == 'SAGE':
                conv = SAGEConv(hidden_channels, hidden_channels)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge attributes (optional)
            
        Returns:
            Enhanced node features (num_nodes, out_channels)
        """
        # Input embedding
        x = self.input_mlp(x)
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + x_new
        
        # Output prediction
        out = self.output_mlp(x)
        
        return out


class MeshEncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture with skip connections for mesh super-resolution.
    Similar to U-Net but adapted for irregular meshes using GNNs.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 4,
                 num_levels: int = 3,
                 dropout: float = 0.1):
        """
        Initialize encoder-decoder.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Base hidden dimension
            out_channels: Number of output features
            num_levels: Number of encoder/decoder levels
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        
        curr_channels = in_channels
        for i in range(num_levels):
            out_ch = hidden_channels * (2 ** i)
            self.encoders.append(nn.Sequential(
                nn.Linear(curr_channels, out_ch),
                nn.LayerNorm(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                GATConv(out_ch, out_ch // 8, heads=8, dropout=dropout)
            ))
            self.encoder_norms.append(nn.LayerNorm(out_ch))
            curr_channels = out_ch
        
        # Bottleneck
        bottleneck_channels = hidden_channels * (2 ** (num_levels - 1))
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels * 2),
            nn.LayerNorm(bottleneck_channels * 2),
            nn.ReLU(),
            nn.Linear(bottleneck_channels * 2, bottleneck_channels),
            nn.LayerNorm(bottleneck_channels),
            nn.ReLU()
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        
        for i in range(num_levels - 1, -1, -1):
            in_ch = hidden_channels * (2 ** i)
            out_ch = hidden_channels * (2 ** (i - 1)) if i > 0 else hidden_channels
            
            # Skip connection doubles the input channels
            self.decoders.append(nn.Sequential(
                nn.Linear(in_ch * 2, out_ch),
                nn.LayerNorm(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                GATConv(out_ch, out_ch // 8, heads=8, dropout=dropout)
            ))
            self.decoder_norms.append(nn.LayerNorm(out_ch))
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity
            
        Returns:
            Enhanced features (num_nodes, out_channels)
        """
        # Encoder with skip connections
        skip_connections = []
        
        for encoder, norm in zip(self.encoders, self.encoder_norms):
            # Apply linear + GNN
            x_linear = encoder[0:4](x)  # Linear + Norm + ReLU + Dropout
            x_gnn = encoder[4](x_linear, edge_index)  # GNN layer
            x = norm(x_gnn)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, (decoder, norm) in enumerate(zip(self.decoders, self.decoder_norms)):
            # Concatenate skip connection
            skip = skip_connections[-(i + 1)]
            x = torch.cat([x, skip], dim=-1)
            
            # Apply decoder
            x_linear = decoder[0:4](x)
            x_gnn = decoder[4](x_linear, edge_index)
            x = norm(x_gnn)
        
        # Output
        out = self.output_head(x)
        
        return out


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline for comparison.
    Operates on each node independently without graph structure.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 256,
                 out_channels: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize MLP.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension
            out_channels: Number of output features
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        curr_channels = in_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(curr_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_channels = hidden_channels
        
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (edge_index ignored for compatibility)."""
        return self.network(x)


def build_knn_graph(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Build k-nearest neighbor graph from mesh coordinates.
    
    Args:
        coords: Node coordinates (num_nodes, ndim)
        k: Number of nearest neighbors
        
    Returns:
        Edge index array (2, num_edges)
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)  # +1 because first neighbor is self
    
    # Build edge list (exclude self-loops)
    edges = []
    for i in range(len(coords)):
        for j in indices[i, 1:]:  # Skip first (self)
            edges.append([i, j])
    
    edge_index = np.array(edges).T
    return edge_index


def create_graph_data(
    coords: np.ndarray,
    features: np.ndarray,
    target: Optional[np.ndarray] = None,
    k: int = 8,
    edge_index: Optional[np.ndarray] = None,
) -> Data:
    """
    Create PyTorch Geometric Data object from mesh data.
    
    Args:
        coords: Node coordinates (num_nodes, ndim)
        features: Node features (num_nodes, num_features)
        target: Target values (optional)
        k: Number of neighbors for graph construction
        
    Returns:
        PyTorch Geometric Data object
    """
    # Build graph connectivity (or reuse precomputed)
    if edge_index is None:
        edge_index = build_knn_graph(coords, k=k)
    
    # Concatenate coordinates with features
    x = np.concatenate([coords, features], axis=-1)
    
    # Convert to tensors
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    
    if target is not None:
        data.y = torch.FloatTensor(target)
    
    return data


if __name__ == "__main__":
    # Test model creation
    print("Testing GNN models...")
    
    # Create dummy data
    num_nodes = 1000
    in_channels = 5  # 2D coords + 3 fields (vx, vy, p)
    out_channels = 4
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 8))
    
    # Test MeshGNN
    model = MeshGNN(in_channels, hidden_channels=64, out_channels=out_channels, num_layers=3)
    out = model(x, edge_index)
    print(f"MeshGNN output shape: {out.shape}")
    
    # Test MLP
    mlp = SimpleMLP(in_channels, hidden_channels=128, out_channels=out_channels)
    out_mlp = mlp(x)
    print(f"MLP output shape: {out_mlp.shape}")
    
    print("Model tests completed successfully!")
