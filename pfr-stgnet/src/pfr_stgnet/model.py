"""The three architectural components described in the manuscript.

One call processes one trajectory window. Gradient accumulation supplies the
effective batch size without materializing 16 x 101 x 1870 residue graphs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RelationGraphLayer(nn.Module):
    def __init__(self, width: int, relations: int = 3, dropout: float = 0.1):
        super().__init__()
        self.messages = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * width + 4, width), nn.GELU(), nn.Linear(width, width))
             for _ in range(relations)]
        )
        self.self_map = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pos, edge_index, edge_type):
        src, dst = edge_index
        delta = pos[src] - pos[dst]
        geometry = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), dim=-1)
        aggregate = torch.zeros_like(h)
        for relation, transform in enumerate(self.messages):
            mask = edge_type == relation
            if mask.any():
                target = dst[mask]
                msg = transform(torch.cat((h[src[mask]], h[target], geometry[mask]), dim=-1))
                partial = torch.zeros_like(h)
                counts = h.new_zeros(h.shape[0], 1)
                partial.index_add_(0, target, msg)
                counts.index_add_(0, target, torch.ones_like(msg[:, :1]))
                aggregate = aggregate + partial / counts.clamp_min(1)
        return self.norm(h + self.dropout(F.gelu(self.self_map(h) + aggregate)))


def _coarse_edges(device):
    src, dst, kind = [], [], []
    for i in range(20):
        for j in range(20):
            if i == j:
                continue
            if i // 4 == j // 4:
                relation = 1
            elif abs(i // 4 - j // 4) == 1:
                relation = 2
            else:
                continue
            src.append(i)
            dst.append(j)
            kind.append(relation)
    return (torch.tensor([src, dst], dtype=torch.long, device=device),
            torch.tensor(kind, dtype=torch.long, device=device))


class PHIE(nn.Module):
    """Three residue message-passing layers, attention pooling, two coarse layers."""

    def __init__(self, input_dim: int = 9, width: int = 128):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.subdomain_embedding = nn.Embedding(20, width)
        self.residue_layers = nn.ModuleList([RelationGraphLayer(width) for _ in range(3)])
        self.pool_score = nn.Linear(width, 1)
        self.coarse_layers = nn.ModuleList([RelationGraphLayer(width) for _ in range(2)])
        coarse_index, coarse_type = _coarse_edges("cpu")
        self.register_buffer("coarse_index", coarse_index)
        self.register_buffer("coarse_type", coarse_type)

    def forward(self, x, pos, edge_index, edge_type, subdomain):
        length, residues, _ = x.shape
        h = self.input(x) + self.subdomain_embedding(subdomain)[None]
        h = h.reshape(length * residues, -1)
        flat_pos = pos.reshape(length * residues, 3)
        for layer in self.residue_layers:
            h = layer(h, flat_pos, edge_index, edge_type)
        h = h.reshape(length, residues, -1)
        pooled, centers = [], []
        score = self.pool_score(h).squeeze(-1)
        for group in range(20):
            members = subdomain == group
            weights = score[:, members].softmax(dim=1)
            pooled.append((weights[..., None] * h[:, members]).sum(dim=1))
            centers.append(pos[:, members].mean(dim=1))
        coarse_h = torch.stack(pooled, dim=1)
        coarse_pos = torch.stack(centers, dim=1)
        offsets = torch.arange(length, device=x.device)[:, None, None] * 20
        index = (self.coarse_index[None] + offsets).permute(1, 0, 2).reshape(2, -1)
        types = self.coarse_type.repeat(length)
        coarse_h = coarse_h.reshape(length * 20, -1)
        coarse_pos = coarse_pos.reshape(length * 20, 3)
        for layer in self.coarse_layers:
            coarse_h = layer(coarse_h, coarse_pos, index, types)
        return coarse_h.reshape(length, 20, -1)


class CausalFiLMBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.film = nn.Linear(32, 2 * width)

    def _causal(self, conv, x):
        return conv(F.pad(x, (2 * self.dilation, 0)))

    def forward(self, x, condition):
        z = self._causal(self.conv1, x)
        z = F.gelu(self.norm1(z.transpose(1, 2))).transpose(1, 2)
        z = self._causal(self.conv2, z)
        z = self.norm2(z.transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.film(condition).chunk(2)
        z = z * (1 + gamma.tanh()[None, :, None]) + beta[None, :, None]
        return x + F.gelu(z)


class FCTB(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        self.blocks = nn.ModuleList([CausalFiLMBlock(width, d) for d in (1, 2, 4, 8, 16)])

    def forward(self, history, force_pn, configuration):
        force = torch.as_tensor(force_pn, dtype=history.dtype, device=history.device).reshape(1)
        config = torch.as_tensor(configuration, dtype=torch.long, device=history.device).reshape(1)
        metadata = torch.cat((force / 250.0, F.one_hot(config, 2).reshape(2).to(history.dtype)))
        condition = self.condition(metadata)
        z = history.permute(1, 2, 0)
        for block in self.blocks:
            z = block(z, condition)
        return z[:, :, -1]  # [20, 128]


class ZARD(nn.Module):
    def __init__(self, width: int, targets: int, horizons: int):
        super().__init__()
        def heads():
            return nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
                              nn.Linear(width, 2 * targets)) for _ in range(horizons)
            ])
        self.background = heads()
        self.residual = heads()

    def forward(self, loaded, zero):
        loaded, zero = loaded.mean(dim=0), zero.mean(dim=0)
        outputs, residuals = [], []
        for background, residual in zip(self.background, self.residual):
            delta = residual(loaded) - residual(zero)
            outputs.append(background(zero) + delta)
            residuals.append(delta)
        output = torch.stack(outputs)
        residual = torch.stack(residuals)
        mean, logvar = output.chunk(2, dim=-1)
        return mean, logvar.clamp(-8, 6), residual.chunk(2, dim=-1)[0]


class PFRSTGNet(nn.Module):
    """Forward forecast + strictly force-blind inverse branch.

    The zero-force residual is identically zero because R(z0)-R(z0) cancels.
    """

    def __init__(self, input_dim: int = 9, targets: int = 74, horizons: int = 3,
                 width: int = 128):
        super().__init__()
        self.phie = PHIE(input_dim, width)
        self.fctb = FCTB(width)
        self.zard = ZARD(width, targets, horizons)
        self.time_attention = nn.Linear(width, 1)
        self.force_head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.configuration_head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, sample, mask_interface: int | None = None):
        edge_index, edge_type = sample["edge_index"], sample["edge_type"]
        if mask_interface is not None:
            keep = sample["edge_interface"] != mask_interface
            edge_index, edge_type = edge_index[:, keep], edge_type[keep]
        history = self.phie(sample["x"], sample["pos"], edge_index, edge_type,
                            sample["subdomain"])
        # This summary is computed before and independently of F and m.
        blind_frames = history.mean(dim=1)
        temporal_weights = self.time_attention(blind_frames).squeeze(-1).softmax(dim=0)
        blind = (temporal_weights[:, None] * blind_frames).sum(dim=0)
        force_hat = self.force_head(blind).squeeze(-1) * 250.0
        configuration_logits = self.configuration_head(blind)
        loaded = self.fctb(history, sample["force_pn"], sample["configuration"])
        zero = self.fctb(history, 0.0, sample["configuration"])
        mean, logvar, residual_mean = self.zard(loaded, zero)
        return {"mean": mean, "logvar": logvar, "residual_mean": residual_mean,
                "force_hat": force_hat, "configuration_logits": configuration_logits}
