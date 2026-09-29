"""Trajectory-level splitting and leakage-free temporal windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .preprocess import TWIST_INDICES


@dataclass(frozen=True)
class Record:
    trajectory_id: str
    state_id: str
    configuration: int
    force_pn: float
    path: Path


def read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [Record(row["trajectory_id"], row["state_id"], int(row["configuration"]),
                       float(row["force_pn"]), Path(row["path"])) for row in csv.DictReader(handle)]


def split_records(records, test_fold: int, mode: str = "state"):
    states = sorted({record.state_id for record in records})
    test_state = states[test_fold % len(states)]
    validation_state = states[(test_fold + 1) % len(states)]
    train = [r for r in records if r.state_id not in (test_state, validation_state)]
    validation = [r for r in records if r.state_id == validation_state]
    test = [r for r in records if r.state_id == test_state]
    if mode == "interpolate":
        train = [r for r in train if r.force_pn != 100]
        validation = [r for r in validation if r.force_pn != 100]
        test = [r for r in test if r.force_pn == 100]
    elif mode == "extrapolate":
        train = [r for r in train if r.force_pn <= 150]
        validation = [r for r in validation if r.force_pn <= 150]
        test = [r for r in test if r.force_pn == 250]
    return train, validation, test


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray
    metric_scale: np.ndarray

    @classmethod
    def fit(cls, records):
        total = None
        total_squared = None
        count = 0
        for record in records:
            with np.load(record.path) as z:
                y = z["y"].astype(np.float64)
            if total is None:
                total = np.zeros(y.shape[-1], np.float64)
                total_squared = np.zeros_like(total)
            total += y.sum(axis=0)
            total_squared += (y * y).sum(axis=0)
            count += len(y)
        mean = total / count
        scale = np.sqrt(np.maximum(total_squared / count - mean * mean, 1e-8))
        metric_scale = scale.copy()
        mean[TWIST_INDICES] = 0
        scale[TWIST_INDICES] = 1
        return cls(mean.astype(np.float32), scale.astype(np.float32),
                   metric_scale.astype(np.float32))

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean


class WindowDataset:
    """One trajectory cached at a time; iterate records before temporal windows."""

    def __init__(self, records, normalizer, history_ns=1.0,
                 horizons_ns=(0.25, 0.5, 1.0), stride=1):
        self.records = list(records)
        self.normalizer = normalizer
        self.history_ns = history_ns
        self.horizons_ns = horizons_ns
        self.stride = stride
        self.zero = {(r.state_id, r.configuration): r for r in records if r.force_pn == 0}
        self.cached_path = None
        self.cached = None
        self.zero_cached_path = None
        self.zero_cached = None

    @staticmethod
    def _read(path):
        with np.load(path) as z:
            return {key: z[key] for key in z.files}

    def _load(self, path, zero=False):
        if zero:
            if path != self.zero_cached_path:
                self.zero_cached = self._read(path)
                self.zero_cached_path = path
            return self.zero_cached
        if path != self.cached_path:
            self.cached = self._read(path)
            self.cached_path = path
        return self.cached

    def window_indices(self, record):
        z = self._load(record.path)
        dt_ps = float(np.median(np.diff(z["time_ps"])))
        history = round(self.history_ns * 1000 / dt_ps) + 1
        horizons = [round(ns * 1000 / dt_ps) for ns in self.horizons_ns]
        return history, horizons, range(history - 1, len(z["time_ps"]) - max(horizons), self.stride)

    def get(self, record, end_frame, device="cpu"):
        z = self._load(record.path)
        history, horizons, _ = self.window_indices(record)
        first = end_frame - history + 1
        x = z["x"][first:end_frame + 1]
        pos = z["pos"][first:end_frame + 1]
        n = x.shape[1]
        src, dst, kind, interface = [], [], [], []
        ptr = z["edge_ptr"]
        for local_t, frame in enumerate(range(first, end_frame + 1)):
            start, stop = ptr[frame:frame + 2]
            src.append(z["edge_src"][start:stop].astype(np.int64) + local_t * n)
            dst.append(z["edge_dst"][start:stop].astype(np.int64) + local_t * n)
            kind.append(z["edge_type"][start:stop])
            interface.append(z["edge_interface"][start:stop])
        y = z["y"][[end_frame + h for h in horizons]]
        reference = self.zero.get((record.state_id, record.configuration))
        if reference is not None:
            zero_z = self._load(reference.path, zero=True)
            zero_y = zero_z["y"][[end_frame + h for h in horizons]]
        else:
            zero_y = np.zeros_like(y)
        sample = {
            "x": torch.as_tensor(x, dtype=torch.float32, device=device),
            "pos": torch.as_tensor(pos, dtype=torch.float32, device=device),
            "subdomain": torch.as_tensor(z["subdomain"], dtype=torch.long, device=device),
            "edge_index": torch.as_tensor(np.stack((np.concatenate(src), np.concatenate(dst))),
                                          dtype=torch.long, device=device),
            "edge_type": torch.as_tensor(np.concatenate(kind), dtype=torch.long, device=device),
            "edge_interface": torch.as_tensor(np.concatenate(interface), dtype=torch.long, device=device),
            "y": torch.as_tensor(self.normalizer.transform(y), dtype=torch.float32, device=device),
            "zero_y": torch.as_tensor(self.normalizer.transform(zero_y), dtype=torch.float32, device=device),
            "last_y": torch.as_tensor(self.normalizer.transform(z["y"][end_frame]),
                                      dtype=torch.float32, device=device),
            "force_pn": torch.tensor(record.force_pn, dtype=torch.float32, device=device),
            "configuration": torch.tensor(record.configuration, dtype=torch.long, device=device),
            "paired": reference is not None,
        }
        return sample
