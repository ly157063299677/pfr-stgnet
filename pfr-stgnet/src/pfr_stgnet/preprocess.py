"""Convert per-monomer C-alpha XVG trajectories into PFR-STGNet graph data.

The supplied pilot folders already contain ca_ACT*.xvg. For new MD runs,
export the same five C-alpha groups with `gmx traj -ox` at the desired cadence.
All lengths below are in nm; XVG time is in ps.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


MONOMERS = ("ACT3", "ACT5", "ACT1", "ACT4", "ACT2")  # pointed -> barbed
TARGET_NAMES = (
    [f"rise_{a}_{b}" for a, b in zip(MONOMERS[:-1], MONOMERS[1:])]
    + [f"twist_{a}_{b}" for a, b in zip(MONOMERS[:-1], MONOMERS[1:])]
    + [f"contact_change_{a}_{b}" for a, b in zip(MONOMERS[:-1], MONOMERS[1:])]
    + [f"displacement_{m}_SD{s}_{axis}"
       for m in MONOMERS for s in range(1, 5) for axis in "xyz"]
    + ["global_axial_change", "global_transverse_change"]
)
TWIST_INDICES = np.arange(4, 8)


def read_xvg(path: Path):
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() and line[0] not in "#@&":
                rows.append(np.fromstring(line, sep=" "))
    array = np.stack(rows)
    return array[:, 0].astype(np.float32), array[:, 1:].reshape(len(rows), -1, 3).astype(np.float32)


def subdomain_for_order(order: np.ndarray, count: int):
    if count == 374:
        # Conventional actin subdomains; positions are zero-based within a monomer.
        return np.select(
            [(order <= 31) | ((order >= 69) & (order <= 143)) | (order >= 337),
             (order >= 32) & (order <= 68),
             ((order >= 144) & (order <= 179)) | ((order >= 269) & (order <= 336)),
             (order >= 180) & (order <= 268)],
            [0, 1, 2, 3], default=0,
        ).astype(np.int64)
    # Synthetic systems use four sites per monomer, one per subdomain.
    return np.floor(4 * order / count).clip(0, 3).astype(np.int64)


def intrinsic_coordinates(coords: np.ndarray, monomer: np.ndarray, subdomain: np.ndarray):
    """Use the initial frame, preserving subsequent axial and transverse motion.

    A rigid transform applied to the whole trajectory transforms the initial
    basis with it, leaving all local coordinates unchanged.
    """
    initial = coords[0]
    centers = np.stack([initial[monomer == m].mean(axis=0) for m in range(5)])
    origin = centers[2]
    longitudinal = centers[4] - centers[0]
    longitudinal /= np.linalg.norm(longitudinal)
    radial = initial[subdomain == 9].mean(axis=0) - origin  # ACT1, SD2
    radial -= np.dot(radial, longitudinal) * longitudinal
    radial /= np.linalg.norm(radial)
    cross = np.cross(longitudinal, radial)
    basis = np.stack((longitudinal, radial, cross), axis=1)
    return (coords - origin) @ basis


def _contact_pairs(xyz, monomer, order, cutoff):
    cells = np.floor(xyz / cutoff).astype(np.int32)
    buckets = {}
    for i, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(i)
    offsets = list(product((-1, 0, 1), repeat=3))
    pairs = []
    cutoff_squared = cutoff * cutoff
    for i, cell in enumerate(cells):
        for offset in offsets:
            bucket = (int(cell[0] + offset[0]), int(cell[1] + offset[1]),
                      int(cell[2] + offset[2]))
            for j in buckets.get(bucket, ()):
                if j <= i or abs(int(monomer[i]) - int(monomer[j])) > 1:
                    continue
                if monomer[i] == monomer[j] and abs(int(order[i]) - int(order[j])) <= 1:
                    continue
                difference = xyz[i] - xyz[j]
                if float(difference @ difference) <= cutoff_squared:
                    relation = 1 if monomer[i] == monomer[j] else 2
                    interface = min(int(monomer[i]), int(monomer[j])) if relation == 2 else -1
                    pairs.append((i, j, relation, interface))
    return pairs


def make_graphs(coords, monomer, order, cutoff=0.8):
    """Three directed relations: sequence, intramonomer, intermonomer."""
    seq = [(i, i + 1, 0, -1) for i in range(len(monomer) - 1)
           if monomer[i] == monomer[i + 1] and order[i + 1] == order[i] + 1]
    src_all, dst_all, type_all, interface_all, pointers = [], [], [], [], [0]
    for xyz in coords:
        for i, j, relation, interface in seq + _contact_pairs(xyz, monomer, order, cutoff):
            src_all.extend((i, j))
            dst_all.extend((j, i))
            type_all.extend((relation, relation))
            interface_all.extend((interface, interface))
        pointers.append(len(src_all))
    return (np.asarray(pointers, dtype=np.int64),
            np.asarray(src_all, dtype=np.int32), np.asarray(dst_all, dtype=np.int32),
            np.asarray(type_all, dtype=np.int8), np.asarray(interface_all, dtype=np.int8))


def descriptors(local, monomer, subdomain, edge_ptr, edge_type, edge_interface):
    frames = len(local)
    centers = np.stack([local[:, monomer == m].mean(axis=1) for m in range(5)], axis=1)
    domains = np.stack([local[:, subdomain == s].mean(axis=1) for s in range(20)], axis=1)
    rise = np.diff(centers[..., 0], axis=1)
    radial = domains[:, 1::4, 1:3] - centers[:, :, 1:3]
    dot = (radial[:, :-1] * radial[:, 1:]).sum(axis=-1)
    cross = radial[:, :-1, 0] * radial[:, 1:, 1] - radial[:, :-1, 1] * radial[:, 1:, 0]
    twist = np.arctan2(cross, dot)
    contacts = np.zeros((frames, 4), dtype=np.float32)
    for t in range(frames):
        start, stop = edge_ptr[t:t + 2]
        selected = edge_interface[start:stop][edge_type[start:stop] == 2]
        contacts[t] = np.bincount(selected, minlength=4)[:4] / 2
    contact_change = (contacts - contacts[0]) / np.maximum(contacts[0], 1)
    domain_displacement = (domains - domains[0]).reshape(frames, -1)
    terminal = centers[:, 4] - centers[:, 0]
    axial = terminal[:, 0:1] - terminal[0, 0]
    transverse = np.linalg.norm(terminal[:, 1:3] - terminal[0, 1:3], axis=-1, keepdims=True)
    return np.concatenate((rise, twist, contact_change, domain_displacement,
                           axial, transverse), axis=-1).astype(np.float32)


def build_trajectory(coords, time_ps, monomer, subdomain, order, output: Path):
    local = intrinsic_coordinates(coords, monomer, subdomain)
    ptr, src, dst, kind, interface = make_graphs(local, monomer, order)
    delta = np.diff(local, axis=0, prepend=local[:1])
    x = np.concatenate((local / 10.0, delta,
                        np.broadcast_to((monomer / 4)[None, :, None], (*local.shape[:2], 1)),
                        np.broadcast_to(((subdomain % 4) / 3)[None, :, None], (*local.shape[:2], 1)),
                        np.broadcast_to((order / np.maximum(np.bincount(monomer)[monomer] - 1, 1))[None, :, None],
                                        (*local.shape[:2], 1))), axis=-1).astype(np.float32)
    y = descriptors(local, monomer, subdomain, ptr, kind, interface)
    np.savez_compressed(output, x=x, pos=local, y=y, time_ps=time_ps,
                        monomer=monomer, subdomain=subdomain, order=order,
                        edge_ptr=ptr, edge_src=src, edge_dst=dst,
                        edge_type=kind, edge_interface=interface)
    return len(local), len(monomer)


def preprocess_xvg(folder: Path, output: Path):
    pieces, monomer, subdomain, order = [], [], [], []
    for m, name in enumerate(MONOMERS):
        time, xyz = read_xvg(folder / f"ca_{name}.xvg")
        pieces.append(xyz)
        count = xyz.shape[1]
        order.append(np.arange(count, dtype=np.int64))
        monomer.append(np.full(count, m, dtype=np.int64))
        subdomain.append(4 * m + subdomain_for_order(order[-1], count))
    return build_trajectory(np.concatenate(pieces, axis=1), time,
                            np.concatenate(monomer), np.concatenate(subdomain),
                            np.concatenate(order), output)


def append_manifest(path: Path, trajectory: Path, state: str, configuration: int,
                    force_pn: float):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(("trajectory_id", "state_id", "configuration", "force_pn", "path"))
        writer.writerow((trajectory.stem, state, configuration, force_pn, trajectory.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--state", type=str, default="pilot")
    parser.add_argument("--configuration", type=int, choices=(0, 1), default=0)
    parser.add_argument("--force-pn", type=float, default=250.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames, residues = preprocess_xvg(args.folder, args.output)
    if args.manifest:
        append_manifest(args.manifest, args.output, args.state, args.configuration, args.force_pn)
    print(f"Saved {frames} frames, {residues} C-alpha nodes: {args.output}")


if __name__ == "__main__":
    main()
