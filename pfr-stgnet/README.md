# PFR-STGNet

Minimal research implementation of the core method in *PFR-STGNet: Polarity-Aware Force-Conditioned Residual Spatiotemporal Graph Network for F-Actin Mechanical Response Forecasting*.

This repository implements the PHIE residue-to-subdomain encoder, the five-block force-conditioned causal temporal module, the exact zero-force residual cancellation in ZARD, a force-blind inverse branch, heteroscedastic training, group-wise state splits, force interpolation/extrapolation tests, and interface masking. It does **not** contain the 80 MD trajectories or reproduce the numerical tables merely by running this code.

## Install

Use a Python environment with a PyTorch build appropriate for your GPU, then:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Smoke test with a synthetic chain

The included Langevin double-helix is an explicit pipeline test. The manuscript does not specify its synthetic potential and integrator fully, so this generator is **not** asserted to reproduce Fig. 2.

```bash
python -m pfr_stgnet.synthetic --output data/synthetic --states 3 --frames 51
python -m pfr_stgnet.train --manifest data/synthetic/manifest.csv --output runs/smoke --fold 0 --epochs 1 --patience 1 --device cpu --max-windows-per-record 1 --max-eval-windows 1
```

The result is written to `runs/smoke/results.json`, with a checkpoint at `runs/smoke/best.pt`.

## Prepare the provided GROMACS C-alpha exports

Each trajectory folder must contain `ca_ACT3.xvg`, `ca_ACT5.xvg`, `ca_ACT1.xvg`, `ca_ACT4.xvg`, and `ca_ACT2.xvg`. Each row is `time_ps, x1, y1, z1, ...`, as in the provided pilot folders. The code uses the actual XVG sampling interval: the supplied pilot exports are at 50 ps, whereas the manuscript specifies 10 ps. Export new XVGs at 10 ps for manuscript-matched experiments.

```powershell
python -m pfr_stgnet.preprocess --folder '250pN_seed12345' --output data/pilot/250pN_seed12345.npz --manifest data/pilot/manifest.csv --state seed12345 --configuration 0 --force-pn 250
python -m pfr_stgnet.preprocess --folder '250pN_seed23456' --output data/pilot/250pN_seed23456.npz --manifest data/pilot/manifest.csv --state seed23456 --configuration 0 --force-pn 250
```

The pilot manifest is useful for checking preprocessing, **not** for training the paper's comparison: it has only two 250-pN trajectories and lacks the matched 0-pN references, the other force levels, and the reverse loading configuration.

For the full dataset, repeat preprocessing once per trajectory. The manifest columns are `trajectory_id,state_id,configuration,force_pn,path`. Use the same `state_id` for every force and both loading configurations derived from one independently equilibrated initial state. Configuration 0 denotes fixed ACT4/ACT2 and loaded ACT3; configuration 1 denotes the swapped arrangement. Each `(state_id, configuration)` should have its own 0-pN reference. State-group splitting occurs before windows are created.

## Train and evaluate

```bash
python -m pfr_stgnet.train --manifest data/full/manifest.csv --output runs/state_fold0 --fold 0 --mode state --seed 42 --interface-analysis
python -m pfr_stgnet.train --manifest data/full/manifest.csv --output runs/interpolation_fold0 --fold 0 --mode interpolate --seed 42
python -m pfr_stgnet.train --manifest data/full/manifest.csv --output runs/extrapolation_fold0 --fold 0 --mode extrapolate --seed 42
```

Run folds `0..7` and five seeds per fold for the manuscript protocol. `--mode interpolate` removes 100 pN from training and validation; `--mode extrapolate` restricts them to at most 150 pN and evaluates 250 pN. AdamW defaults to learning rate `1e-3`, weight decay `1e-4`, effective batch size 16 via accumulation, and patience 20. `--stride 10` avoids training on every highly overlapping window; set `--stride 1` if exact every-frame windows are needed.

## Target and graph definitions

The preprocessor uses the pointed-to-barbed order ACT3-ACT5-ACT1-ACT4-ACT2. A right-handed reference frame is fixed from the initial configuration, with its longitudinal axis from ACT3 to ACT2, a radial axis from ACT1 center toward ACT1 subdomain 2, and their cross product. Holding this frame fixed preserves later transverse motion while making the whole trajectory invariant to a common rigid rotation and translation. Three directed residue edge relations are built: sequential, intramonomer contacts, and adjacent-intermonomer contacts within 0.8 nm. Four conventional actin subdomains per monomer produce 20 coarse nodes.

The 74-dimensional response vector contains four interface rises (nm), four signed twists (rad), four relative contact-count changes, 20 three-component subdomain-center displacements (nm), and global axial and transverse terminal changes (nm). These are *operational definitions supplied by this repository*: the manuscript names these descriptors but does not specify every atom selection and normalization. Reconcile them with the authors' final analysis protocol before claiming numerical reproduction.

The model uses three 128-channel residue graph layers, attention pooling, two coarse graph layers, and five residual temporal blocks with kernel 3 and dilations 1/2/4/8/16 (125-frame receptive field). The same PHIE history is evaluated under actual and zero force. The predicted mean is `B(Z(0,m)) + R(Z(F,m)) - R(Z(0,m))`; therefore the force-induced term is exactly zero at `F=0`. Inverse heads read only unconditioned PHIE history. The pair loss is active when a matched 0-pN trajectory is present.

## Scope and reproducibility

The paper's reported NRMSE, confidence intervals, baselines, ensemble calibration, and independent-validation findings require the complete trajectories and experiment outputs; they are not embedded in this repository. This code supplies the method and a measurable pipeline, not fabricated replication of the manuscript's tables. Raw MD files and processed arrays are ignored by Git. Share the data separately if permission and storage allow.
