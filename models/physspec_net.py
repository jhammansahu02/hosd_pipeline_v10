"""
PhysSpec-Net v2: Clark et al. 2010 Physics-Exact Architecture
==============================================================

Base: PhysSpecNet v1 (working, F1=0.689)
New:  Clark et al. 2010 physics-motivated improvements

Changes from v1:
  1. Band groups updated to Clark et al. wavelength regions
  2. Shoulderness features added (Clark et al. key diagnostic)
  3. Oil-water ratio features added (Clark et al.)
  4. Raw skip connection added
  5. Classifier: 3*group_dim (avg+max+raw)

Clark et al. 2010 (USGS Report 2010-1167):
  0.93um -> water absorption reference  -> G2 (bands 55-65)
  1.20um -> organic absorption (OIL)    -> G3 (bands 65-100) KEY
  1.70um -> organic absorption (OIL)    -> G4 (bands 100-155) KEY
  2.30um -> C-H absorption (thickness)  -> G5 (bands 155-203)

Params: ~227K
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PHYS_GROUPS = [
    (0,   55),   # G1: Visible (0.365-0.90um) — high oil/water contrast
    (55,  65),   # G2: Water reference (0.93um) — Clark water absorption
    (65, 100),   # G3: Oil absorption 1.2um — Clark KEY diagnostic
    (100, 155),  # G4: Oil peak + 1.7um — Clark KEY diagnostic
    (155, 203),  # G5: C-H absorption 2.3um — oil thickness
]

N_GROUPS    = len(PHYS_GROUPS)
GROUP_DIM   = 64
CROSS_HEADS = 4

# Clark et al. absorption centers (203 good-band indices)
OIL_12_CENTER = 83    # 1.2um oil absorption
OIL_17_CENTER = 133   # 1.7um oil absorption
SHOULDER_WIN  = 5


class GroupEncoder(nn.Module):
    """Same as v1 — proven working."""
    def __init__(self, in_features: int, out_dim: int = GROUP_DIM,
                 dropout: float = 0.3):
        super().__init__()
        hidden = max(in_features, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CrossGroupAttention(nn.Module):
    """Same as v1 — proven working."""
    def __init__(self, d_model: int = GROUP_DIM,
                 n_heads: int = CROSS_HEADS,
                 dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PhysSpecNet(nn.Module):
    """
    PhysSpec-Net v2 — Clark et al. Physics-Exact.

    Tokens (9):
      G1-G5: Clark et al. physics band groups
      deriv:    spectral derivatives
      shoulder: Clark shoulderness at 1.2 + 1.7um
      ratio:    oil/water ratios (Clark diagnostic)
      raw:      skip connection

    Input : (N, in_bands)
    Output: (N, 2)
    Params: ~227K
    """
    def __init__(self, in_bands: int, n_classes: int = 2,
                 group_dim: int = GROUP_DIM,
                 n_cross_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        sys.stderr.write(
            f"PhysSpecNet v2 (Clark et al.) | "
            f"in_bands={in_bands} | group_dim={group_dim}\n")
        sys.stderr.flush()

        self.in_bands = in_bands

        # Group slices
        self.group_slices = []
        for s, e in PHYS_GROUPS:
            s = min(s, in_bands)
            e = min(e, in_bands)
            if e > s:
                self.group_slices.append((s, e))

        # Derivative branch (v1 same)
        deriv_dim = (in_bands - 1) + (in_bands - 2)
        self.deriv_proj = nn.Sequential(
            nn.Linear(deriv_dim, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, group_dim),
            nn.BatchNorm1d(group_dim), nn.GELU(),
        )

        # Group encoders (v1 same)
        self.group_encoders = nn.ModuleList([
            GroupEncoder(e - s, group_dim, dropout)
            for s, e in self.group_slices
        ])

        # Shoulderness projection — Clark et al.
        # 2 values: at 1.2um and 1.7um
        self.shoulder_proj = nn.Sequential(
            nn.Linear(2, group_dim),
            nn.GELU(),
        )

        # Oil-water ratio projection — Clark et al.
        # 3 ratios: G3/G2, G4/G2, G3/G4
        self.ratio_proj = nn.Sequential(
            nn.Linear(3, group_dim),
            nn.GELU(),
        )

        # Raw skip connection
        self.raw_proj = nn.Sequential(
            nn.Linear(in_bands, group_dim),
            nn.GELU(),
        )

        # Cross-group attention (v1 same: 2 layers, 4 heads)
        self.cross_attn_layers = nn.ModuleList([
            CrossGroupAttention(group_dim, CROSS_HEADS, dropout=0.1)
            for _ in range(n_cross_layers)
        ])

        # Classifier: avg + max + raw = 3 * group_dim
        self.classifier = nn.Sequential(
            nn.Linear(3 * group_dim, 128),  # 3*64=192 -> 128
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

        # Count and print params
        n_params = sum(p.numel() for p in self.parameters()
                       if p.requires_grad)
        sys.stderr.write(f"PhysSpecNet v2 params: {n_params:,}\n")
        sys.stderr.flush()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _shoulderness(self, x: torch.Tensor,
                      center: int) -> torch.Tensor:
        """Clark et al. shoulderness: slope change around absorption."""
        c = min(max(center, SHOULDER_WIN),
                self.in_bands - SHOULDER_WIN - 1)
        left  = x[:, c - SHOULDER_WIN:c].mean(dim=1, keepdim=True)
        right = x[:, c:c + SHOULDER_WIN].mean(dim=1, keepdim=True)
        return torch.clamp(right - left, -5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Derivatives (v1 same)
        d1 = x[:, 1:] - x[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        deriv_emb = self.deriv_proj(torch.cat([d1, d2], dim=1))

        # Group encoders (v1 same)
        group_embs = [
            enc(x[:, s:e])
            for (s, e), enc in zip(self.group_slices,
                                    self.group_encoders)
        ]

        # Shoulderness — Clark et al.
        s1 = self._shoulderness(x, OIL_12_CENTER)  # 1.2um
        s2 = self._shoulderness(x, OIL_17_CENTER)  # 1.7um
        shoulder_emb = self.shoulder_proj(
            torch.cat([s1, s2], dim=1))

        # Oil-water ratios — Clark et al.
        g2 = x[:, self.group_slices[1][0]:
                   self.group_slices[1][1]].mean(1, keepdim=True)
        g3 = x[:, self.group_slices[2][0]:
                   self.group_slices[2][1]].mean(1, keepdim=True)
        g4 = x[:, self.group_slices[3][0]:
                   self.group_slices[3][1]].mean(1, keepdim=True)
        ratios = torch.clamp(torch.cat([
            g3 / (g2.abs() + 1.0),   # oil(1.2um)/water
            g4 / (g2.abs() + 1.0),   # oil(1.7um)/water
            g3 / (g4.abs() + 1.0),   # 1.2um/1.7um ratio
        ], dim=1), -5, 5)
        ratio_emb = self.ratio_proj(ratios)

        # Raw skip
        raw_emb = self.raw_proj(x)

        # Stack 9 tokens
        tokens = torch.stack(
            group_embs + [deriv_emb, shoulder_emb,
                          ratio_emb, raw_emb], dim=1)

        # Cross-group attention (v1 same)
        for layer in self.cross_attn_layers:
            tokens = layer(tokens)

        # Classifier: avg + max + raw
        avg_pool = tokens.mean(dim=1)         # (N, 64)
        max_pool = tokens.max(dim=1).values   # (N, 64)
        feat     = torch.cat([avg_pool, max_pool, raw_emb],
                              dim=1)          # (N, 192)
        return self.classifier(feat)


def get_physspec_net(in_bands: int,
                     n_classes: int = 2) -> PhysSpecNet:
    return PhysSpecNet(in_bands=in_bands, n_classes=n_classes)
