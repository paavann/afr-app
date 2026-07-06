"""
UV-Net segmentation model.

Implements the face segmentation model from 
"UV-Net: Learning from Boundary Representations" (CVPR 2021).

Architecture:
  1. UVNetSurfaceEncoder — 2D CNN on per-face UV-grids → face embeddings
  2. UVNetCurveEncoder  — 1D CNN on per-edge U-grids  → edge embeddings
  3. UVNetGraphEncoder  — GNN (NNConv) message passing on face-adjacency graph
  4. _NonLinearClassifier — 3-layer MLP for per-node classification
"""
import logging

import torch
from torch import nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    PL_AVAILABLE = True
except ImportError:
    try:
        import lightning.pytorch as pl
        PL_AVAILABLE = True
    except ImportError:
        PL_AVAILABLE = False

from uvnet.encoders import (
    UVNetCurveEncoder,
    UVNetSurfaceEncoder,
    UVNetGraphEncoder,
)

logger = logging.getLogger(__name__)


class _NonLinearClassifier(nn.Module):
    """3-layer MLP classifier with BatchNorm and dropout."""

    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self._init_weights(m)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        return self.linear3(x)


class UVNetSegmenter(nn.Module):
    """
    UV-Net face segmentation model (pure nn.Module).

    Takes a DGL graph with face UV-grid features (ndata['x']) and
    edge U-grid features (edata['x']), returns per-node logits.
    """

    def __init__(
        self,
        num_classes=7,
        crv_in_channels=6,
        srf_in_channels=7,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        num_gnn_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.curv_encoder = UVNetCurveEncoder(crv_in_channels, crv_emb_dim)
        self.surf_encoder = UVNetSurfaceEncoder(srf_in_channels, srf_emb_dim)
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim=srf_emb_dim,
            crv_emb_dim=crv_emb_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=num_gnn_layers,
        )
        self.classifier = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, graph):
        # Encode per-face UV-grids
        face_feats = graph.ndata["x"]  # [N, C_srf, H, W]
        face_emb = self.surf_encoder(face_feats)  # [N, srf_emb_dim]

        # Encode per-edge U-grids
        edge_feats = graph.edata["x"]  # [E, C_crv, L]
        edge_emb = self.curv_encoder(edge_feats)  # [E, crv_emb_dim]

        # GNN message passing on face-adjacency graph
        node_emb = self.graph_encoder(graph, face_emb, edge_emb)  # [N, graph_emb_dim]

        # Per-node classification
        logits = self.classifier(node_emb)  # [N, num_classes]
        return logits


if PL_AVAILABLE:
    class Segmentation(pl.LightningModule):
        """
        PyTorch Lightning wrapper for UV-Net face segmentation.
        
        Supports loading from Lightning checkpoints via `load_from_checkpoint()`.
        """

        def __init__(
            self,
            num_classes=7,
            crv_in_channels=6,
            srf_in_channels=7,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            num_gnn_layers=2,
            dropout=0.3,
            lr=1e-3,
            **kwargs,
        ):
            super().__init__()
            self.save_hyperparameters()
            self.model = UVNetSegmenter(
                num_classes=num_classes,
                crv_in_channels=crv_in_channels,
                srf_in_channels=srf_in_channels,
                crv_emb_dim=crv_emb_dim,
                srf_emb_dim=srf_emb_dim,
                graph_emb_dim=graph_emb_dim,
                num_gnn_layers=num_gnn_layers,
                dropout=dropout,
            )
            self.lr = lr

        def forward(self, graph):
            return self.model(graph)

        def training_step(self, batch, batch_idx):
            graph, labels = batch
            logits = self(graph)
            loss = F.cross_entropy(logits, labels)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            graph, labels = batch
            logits = self(graph)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

else:
    # Fallback: no PyTorch Lightning available — use nn.Module directly
    class Segmentation(nn.Module):
        """
        Fallback UV-Net segmentation model (no Lightning).
        Uses torch.load for checkpoint loading.
        """

        def __init__(self, num_classes=7, **kwargs):
            super().__init__()
            self.model = UVNetSegmenter(num_classes=num_classes, **kwargs)

        def forward(self, graph):
            return self.model(graph)

        @classmethod
        def load_from_checkpoint(cls, ckpt_path, map_location=None, **kwargs):
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

            # Extract hyperparameters
            hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
            num_classes = hparams.get("num_classes", 7)

            instance = cls(num_classes=num_classes, **{k: v for k, v in hparams.items() if k != "num_classes"})

            # Load state dict
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Strip 'model.' prefix if present (Lightning convention)
            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.model.", "model.").replace("module.", "")
                cleaned[new_key] = v

            instance.load_state_dict(cleaned, strict=False)
            logger.info("Loaded UV-Net checkpoint (num_classes=%d)", num_classes)
            return instance
