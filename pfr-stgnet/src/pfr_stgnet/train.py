"""Train/evaluate one state-held-out fold of PFR-STGNet."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import Normalizer, WindowDataset, read_manifest, split_records
from .losses import objective
from .metrics import balanced_accuracy, prediction_error, trajectory_metrics
from .model import PFRSTGNet


def evaluate(model, dataset, device, max_windows=None, mask_interface=None):
    model.eval()
    trajectories = []
    all_force_true, all_force_pred, all_cfg_true, all_cfg_pred = [], [], [], []
    with torch.no_grad():
        for record in dataset.records:
            _, _, ends = dataset.window_indices(record)
            ends = list(ends)
            if max_windows:
                ends = ends[:max_windows]
            squared = []
            for end in ends:
                sample = dataset.get(record, end, device)
                output = model(sample, mask_interface=mask_interface)
                squared.append(prediction_error(output["mean"].cpu().numpy(),
                                                sample["y"].cpu().numpy(), dataset.normalizer))
                all_force_true.append(record.force_pn)
                all_force_pred.append(float(output["force_hat"].cpu()))
                if record.force_pn != 0:
                    all_cfg_true.append(record.configuration)
                    all_cfg_pred.append(int(output["configuration_logits"].argmax().cpu()))
            if squared:
                trajectories.append({"trajectory_id": record.trajectory_id,
                                     "state_id": record.state_id, **trajectory_metrics(squared)})
    horizon_scores = np.asarray([row["nrmse_by_horizon"] for row in trajectories])
    result = {"mean_nrmse": float(horizon_scores.mean()),
              "nrmse_by_horizon": horizon_scores.mean(axis=0).tolist(),
              "force_mae_pn": float(np.mean(np.abs(np.asarray(all_force_true) - all_force_pred))),
              "configuration_balanced_accuracy": balanced_accuracy(all_cfg_true, all_cfg_pred),
              "trajectories": trajectories}
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = read_manifest(args.manifest)
    train_records, validation_records, test_records = split_records(records, args.fold, args.mode)
    normalizer = Normalizer.fit(train_records)
    dataset_args = (normalizer, args.history_ns, tuple(args.horizons_ns), args.stride)
    training = WindowDataset(train_records, *dataset_args)
    validation = WindowDataset(validation_records, *dataset_args)
    test = WindowDataset(test_records, *dataset_args)
    model = PFRSTGNet(targets=len(normalizer.mean), horizons=len(args.horizons_ns),
                      width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        shuffled = training.records[:]
        random.shuffle(shuffled)
        running, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for record in shuffled:
            _, _, ends = training.window_indices(record)
            ends = list(ends)
            random.shuffle(ends)
            if args.max_windows_per_record:
                ends = ends[:args.max_windows_per_record]
            for end in ends:
                sample = training.get(record, end, device)
                loss, _ = objective(model(sample), sample)
                (loss / args.accumulate).backward()
                running += float(loss.detach().cpu())
                steps += 1
                if steps % args.accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if steps % args.accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        valid = evaluate(model, validation, device, args.max_eval_windows)
        print(f"epoch={epoch + 1} loss={running / steps:.4f} val_nrmse={valid['mean_nrmse']:.4f}")
        if valid["mean_nrmse"] < best:
            best = valid["mean_nrmse"]
            stale = 0
            torch.save({"model": model.state_dict(), "normalizer": normalizer,
                        "args": vars(args), "validation": valid}, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"fold": args.fold, "mode": args.mode, "seed": args.seed,
              "train_states": sorted({r.state_id for r in train_records}),
              "validation_states": sorted({r.state_id for r in validation_records}),
              "test_states": sorted({r.state_id for r in test_records}),
              "validation": checkpoint["validation"],
              "test": evaluate(model, test, device, args.max_eval_windows),
              "parameters": sum(p.numel() for p in model.parameters())}
    if args.interface_analysis:
        result["interface_mask_nrmse"] = [
            evaluate(model, test, device, args.max_eval_windows, mask_interface=i)["mean_nrmse"]
            for i in range(4)]
        result["interface_importance"] = [score - result["test"]["mean_nrmse"]
                                          for score in result["interface_mask_nrmse"]]
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("fold", "mode", "parameters", "test")},
                     indent=2)[:1600])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", choices=("state", "interpolate", "extrapolate"), default="state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--history-ns", type=float, default=1.0)
    parser.add_argument("--horizons-ns", type=float, nargs=3, default=(0.25, 0.5, 1.0))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-windows-per-record", type=int)
    parser.add_argument("--max-eval-windows", type=int)
    parser.add_argument("--interface-analysis", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
