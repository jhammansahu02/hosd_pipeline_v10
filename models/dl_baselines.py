"""
DL Baseline Models — Final
============================
Literature-based DL methods for HSI classification:

1. SpectralCNN1D  — Hu et al. 2015 (1D CNN for HSI)
   DOI: 10.1155/2015/258619

2. HybridSN       — Roy et al. 2020 (3D+2D CNN for HSI)
   DOI: 10.1109/LGRS.2019.2918719

3. SpectralFormer — Hong et al. 2022 (Transformer for HSI)
   DOI: 10.1109/TGRS.2021.3130716
   FIX: Band compression (203→32 tokens) to avoid CUDA
        memory error on Quadro P1000 (4GB VRAM).
        Original SpectralFormer uses one token per band
        (203+1=204 tokens) causing O(B²) attention memory.
        Our adaptation compresses bands into 32 group tokens
        while preserving spectral structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.vit_pytorch import ViT



# ──────────────────────────────────────────────────────────────
class SpectralCNN1D(nn.Module):
    """
    Hu et al. 2015 — 1D spectral CNN baseline.
    Treats spectrum as 1D signal with conv filters.
    """
    def __init__(self, in_bands: int, n_classes: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(128*16, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


# ──────────────────────────────────────────────────────────────
class HybridSN(nn.Module):
    """
    Roy et al. 2020 — HybridSN adapted for pixel-wise classification.
    Uses spectral dimension with 3D+2D style conv hierarchy.
    """
    def __init__(self, in_bands: int, n_classes: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        self.in_bands = in_bands

        self.conv3d_1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.BatchNorm1d(8), nn.GELU(),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.GELU(),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv2d(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
class _SpectralPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# class SpectralFormer(nn.Module):
#     """
#     Hong et al. 2022 — SpectralFormer for HSI pixel classification.

#     ADAPTATION for memory-constrained GPUs:
#     Original: 1 token per band → 203+1=204 tokens → O(B²) attention
#               → CUDA error on Quadro P1000 (4GB VRAM)

#     Our fix:  Band group compression 203 → N_TOKENS=32 group tokens
#               → O(32²) attention = 40x memory reduction
#               → Groups of ~6 adjacent bands averaged per token
#               → Preserves local spectral structure ✓

#     Paper cite: "Following Hong et al. [X], we adapt SpectralFormer
#     with spectral group compression (203→32 tokens) to satisfy
#     GPU memory constraints, reducing quadratic attention complexity
#     from O(B²) to O(T²) where T=32<<B=203."
#     """
#     def __init__(self, in_bands: int, n_classes: int = 2,
#                  d_model: int = 64, n_heads: int = 4,
#                  n_layers: int = 3, dropout: float = 0.3,
#                  n_tokens: int = 32):
#         super().__init__()
#         self.in_bands = in_bands
#         self.d_model  = d_model
#         self.n_tokens = n_tokens

#         # Group size for band compression
#         self.group_size = in_bands // n_tokens      # ~6 bands/group
#         self.n_full     = self.group_size * n_tokens # usable bands
#         # Remainder bands (if any) compressed into last token

#         # Band group embedding: group_size bands → d_model
#         self.band_embed = nn.Sequential(
#             nn.Linear(self.group_size, d_model),
#             nn.LayerNorm(d_model),
#         )

#         # Positional encoding for n_tokens + 1 (CLS)
#         self.pos_enc   = _SpectralPE(d_model, max_len=n_tokens + 1)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

#         # Transformer encoder
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=n_heads,
#             dim_feedforward=d_model * 4,
#             dropout=dropout, batch_first=True,
#             norm_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(
#             enc_layer, num_layers=n_layers)
#         self.norm = nn.LayerNorm(d_model)

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(d_model * 2, n_classes),
#         )
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         # x: (N, B)
#         N, B = x.shape

#         # Compress bands into group tokens
#         # Take first n_full bands, reshape into groups
#         x_full = x[:, :self.n_full]                     # (N, n_full)
#         tokens = x_full.reshape(N, self.n_tokens,
#                                  self.group_size)        # (N, T, G)
#         tokens = self.band_embed(tokens)                 # (N, T, d_model)

#         # Prepend CLS token
#         cls    = self.cls_token.expand(N, -1, -1)       # (N, 1, d_model)
#         tokens = torch.cat([cls, tokens], dim=1)         # (N, T+1, d_model)

#         # Positional encoding + transformer
#         tokens = self.pos_enc(tokens)
#         out    = self.transformer(tokens)
#         out    = self.norm(out)

#         # CLS token → classifier
#         return self.classifier(out[:, 0, :])


# ══════════════════════════════════════════════════════════════
# SpectralFormer — Hong et al. 2022
# ══════════════════════════════════════════════════════════════

class SpectralFormer(nn.Module):
    """
    SpectralFormer — Hong et al. 2022, IEEE TGRS
    Original implementation via vit_pytorch.py
    DOI: 10.1109/TGRS.2021.3130716
    """
    def __init__(self, in_bands, n_classes=2):
        super().__init__()
        self.vit = ViT(
            image_size  = 1,
            near_band   = 3,
            num_patches = in_bands,
            num_classes = n_classes,
            dim         = 64,
            depth       = 5,
            heads       = 4,
            mlp_dim     = 8,
            dropout     = 0.1,
            emb_dropout = 0.1,
            mode        = 'CAF'
        )

    def forward(self, x):
        x_pad = F.pad(x, (1, 1), mode='reflect')
        x_nb  = x_pad.unfold(1, 3, 1)  # (N, B, 3) ✓
        return self.vit(x_nb)
# ──────────────────────────────────────────────────────────────
def get_dl_baselines(in_bands: int, n_classes: int = 2) -> dict:
    """Return all DL baseline models."""
    return {
        "SpectralCNN1D":  SpectralCNN1D(in_bands, n_classes),
        "HybridSN":       HybridSN(in_bands, n_classes),
        "SpectralFormer": SpectralFormer(in_bands, n_classes),
    }
