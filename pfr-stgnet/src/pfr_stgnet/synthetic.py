"""Small, explicit double-helical Langevin benchmark for pipeline testing.

The manuscript does not specify its interaction potential or integrator in
enough detail for exact reproduction. This generator is an illustrative
controlled benchmark, not the source of any number printed in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .preprocess import append_manifest, build_trajectory, subdomain_for_order


def helix():
    indices = np.arange(20)
    layer, strand = indices // 2, indices % 2
    angle = layer * 0.68 + strand * np.pi
    return np.stack((0.26 * layer, 0.32 * np.cos(angle), 0.32 * np.sin(angle)), axis=-1)


def spring_pairs(base):
    pairs = [(i, i + 2) for i in range(18)]
    pairs += [(i, i + 1) for i in range(0, 20, 2)]
    pairs += [(i, i + 3) for i in range(0, 17, 2)]
    lengths = np.asarray([np.linalg.norm(base[j] - base[i]) for i, j in pairs])
    return pairs, lengths


def simulate(initial, force_pn, configuration, frames, dt_ps, noise, base):
    pairs, rest = spring_pairs(base)
    position = initial.copy()
    velocity = np.zeros_like(position)
    fixed = np.arange(16, 20) if configuration == 0 else np.arange(0, 4)
    loaded = np.arange(0, 4) if configuration == 0 else np.arange(16, 20)
    direction = 1 if configuration == 0 else -1
    output = np.empty((frames, 20, 3), np.float32)
    step = 0.02
    for t in range(frames):
        output[t] = position
        spring_force = np.zeros_like(position)
        for (i, j), distance in zip(pairs, rest):
            delta = position[j] - position[i]
            size = np.linalg.norm(delta)
            pull = 12 * (size - distance) * delta / size
            spring_force[i] += pull
            spring_force[j] -= pull
        spring_force[loaded, 0] += direction * (force_pn / 250) * 0.25
        velocity = 0.92 * velocity + step * spring_force + noise[t] * 0.003
        position = position + step * velocity
        position[fixed] = initial[fixed]
        velocity[fixed] = 0
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--frames", type=int, default=101)
    parser.add_argument("--dt-ps", type=float, default=50.0)
    parser.add_argument("--force-levels", type=float, nargs="+", default=(0, 50, 100, 150, 250))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    if manifest.exists():
        manifest.unlink()
    base = helix()
    monomer = np.repeat(np.arange(5), 4).astype(np.int64)
    order = np.tile(np.arange(4), 5).astype(np.int64)
    subdomain = np.concatenate([4 * m + subdomain_for_order(np.arange(4), 4)
                                for m in range(5)])
    time = np.arange(args.frames, dtype=np.float32) * args.dt_ps
    for state in range(args.states):
        initial = base + np.random.default_rng(args.seed + state).normal(0, 0.015, base.shape)
        for configuration in (0, 1):
            noise = np.random.default_rng(args.seed + 1000 * state + configuration).normal(
                size=(args.frames, 20, 3))
            for force in args.force_levels:
                name = f"state{state:02d}_config{configuration}_force{force:g}"
                path = args.output / f"{name}.npz"
                coords = simulate(initial, force, configuration, args.frames,
                                  args.dt_ps, noise, base)
                build_trajectory(coords, time, monomer, subdomain, order, path)
                append_manifest(manifest, path, f"state{state:02d}", configuration, force)
    print(f"Wrote {args.states * 2 * len(args.force_levels)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
