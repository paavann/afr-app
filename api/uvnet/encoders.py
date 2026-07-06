"""
UV-Net encoder architecture.

Implements the convolutional encoders and graph neural network
from "UV-Net: Learning from Boundary Representations" (CVPR 2021).
"""
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """1D conv block: Conv1d → BatchNorm1d → LeakyReLU"""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """2D conv block: Conv2d → BatchNorm2d → LeakyReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    """FC block: Linear → BatchNorm1d → LeakyReLU"""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    """MLP with linear output (no activation on last layer)."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of MLP layers must be >= 1")

        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)

        for i in range(self.num_layers - 1):
            x = F.relu(self.batch_norms[i](self.linears[i](x)))
        return self.linears[-1](x)


class UVNetCurveEncoder(nn.Module):
    """
    1D CNN encoder for edge UV-grids (U-grids).
    
    Processes the sampled 1D curve features (points + tangents)
    along each B-Rep edge to produce a fixed-dimensional embedding.
    """

    def __init__(self, in_channels=6, output_dims=64):
        super().__init__()
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1)
        self.final = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            _fc(256, output_dims),
        )

    def forward(self, x):
        """
        Args:
            x: [E, C_crv, L] — edge features in channel-first format
        Returns:
            [E, output_dims] — per-edge embeddings
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.final(x)


class UVNetSurfaceEncoder(nn.Module):
    """
    2D CNN encoder for face UV-grids.
    
    Processes the sampled 2D surface features (points + normals + mask)
    on each B-Rep face to produce a fixed-dimensional embedding.
    """

    def __init__(self, in_channels=7, output_dims=64):
        super().__init__()
        self.conv1 = _conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = _conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = _conv2d(128, 256, kernel_size=3, padding=1)
        self.final = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            _fc(256, output_dims),
        )

    def forward(self, x):
        """
        Args:
            x: [N, C_srf, H, W] — face features in channel-first format
        Returns:
            [N, output_dims] — per-face embeddings
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.final(x)


class UVNetGraphEncoder(nn.Module):
    """
    Graph Neural Network encoder using NNConv.
    
    Performs message passing on the face-adjacency graph using
    edge embeddings (from curve encoder) as edge features.
    For segmentation, outputs per-node embeddings.
    """

    def __init__(self, srf_emb_dim=64, crv_emb_dim=64, graph_emb_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = srf_emb_dim if i == 0 else graph_emb_dim
            out_dim = graph_emb_dim
            # Edge network: maps edge embeddings → weight matrix for message passing
            edge_net = _MLP(
                num_layers=2,
                input_dim=crv_emb_dim,
                hidden_dim=graph_emb_dim,
                output_dim=in_dim * out_dim,
            )

            # NOTE: DGL's NNConv defaults to residual=False, meaning a node's
            # own previous-layer embedding is completely discarded and
            # replaced purely by the aggregated neighbor messages. This
            # diverges from the paper's GIN-style update (Eq. 1), which
            # explicitly retains (1+eps)*h_v^(k-1) alongside the neighbor
            # aggregation, and it means any face with no effective in-edges
            # collapses to an identical bias-only embedding. residual=True
            # restores the self-term.
            #
            # IMPORTANT: this changes the model's parameter shapes/behavior.
            # An already-trained checkpoint will NOT have these residual
            # weights and will need to be retrained (or at least fine-tuned)
            # after this change — flipping this flag alone will not improve
            # an existing checkpoint's predictions.
            self.convs.append(
                NNConv(in_dim, out_dim, edge_net, aggregator_type="sum", residual=True)
            )
            self.bns.append(nn.BatchNorm1d(out_dim))

    def forward(self, graph, node_feats, edge_feats):
        """
        Args:
            graph: DGL graph
            node_feats: [N, srf_emb_dim] — per-node features (from surface encoder)
            edge_feats: [E, crv_emb_dim] — per-edge features (from curve encoder)
        Returns:
            [N, graph_emb_dim] — per-node graph embeddings
        """
        x = node_feats
        for conv, bn in zip(self.convs, self.bns):
            x = conv(graph, x, edge_feats)
            x = bn(x)
            x = F.leaky_relu(x)
        return x