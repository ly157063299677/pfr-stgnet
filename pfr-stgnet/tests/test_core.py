import numpy as np
import torch

from pfr_stgnet.data import Record, split_records
from pfr_stgnet.model import CausalFiLMBlock, PFRSTGNet
from pfr_stgnet.preprocess import TARGET_NAMES, build_trajectory, subdomain_for_order


def tiny_sample():
    length, nodes = 5, 20
    src = []
    dst = []
    for t in range(length):
        for i in range(nodes - 1):
            src.extend((t * nodes + i, t * nodes + i + 1))
            dst.extend((t * nodes + i + 1, t * nodes + i))
    edges = torch.tensor([src, dst])
    return {"x": torch.randn(length, nodes, 9), "pos": torch.randn(length, nodes, 3),
            "subdomain": torch.arange(20), "edge_index": edges,
            "edge_type": torch.zeros(len(src), dtype=torch.long),
            "edge_interface": torch.full((len(src),), -1, dtype=torch.long),
            "force_pn": torch.tensor(0.0), "configuration": torch.tensor(0)}


def test_zero_force_anchor_and_force_blind_inverse():
    torch.manual_seed(0)
    model = PFRSTGNet(width=16).eval()
    sample = tiny_sample()
    with torch.no_grad():
        zero = model(sample)
        sample["force_pn"] = torch.tensor(150.0)
        loaded = model(sample)
    assert zero["mean"].shape == (3, len(TARGET_NAMES))
    assert torch.count_nonzero(zero["residual_mean"]) == 0
    assert torch.equal(zero["force_hat"], loaded["force_hat"])
    assert torch.equal(zero["configuration_logits"], loaded["configuration_logits"])


def test_causal_block_cannot_read_future():
    torch.manual_seed(1)
    block = CausalFiLMBlock(8, dilation=2).eval()
    x = torch.randn(20, 8, 12)
    changed = x.clone()
    changed[:, :, 8:] += 100
    condition = torch.randn(32)
    with torch.no_grad():
        a, b = block(x, condition), block(changed, condition)
    assert torch.allclose(a[:, :, :8], b[:, :, :8], atol=1e-6)


def test_state_split_before_force_holdout():
    records = [Record(f"s{s}_f{f}", f"s{s}", 0, f, None)
               for s in range(4) for f in (0, 100, 150, 250)]
    train, valid, test = split_records(records, 0, "interpolate")
    assert not ({r.state_id for r in train} & {r.state_id for r in valid + test})
    assert {r.force_pn for r in test} == {100}
    assert all(r.force_pn != 100 for r in train + valid)


def test_preprocessor_produces_20_nodes_and_74_targets(tmp_path):
    from pfr_stgnet.synthetic import helix
    base = helix().astype(np.float32)
    coords = np.stack((base, base + np.array([0.01, 0, 0], np.float32)))
    monomer = np.repeat(np.arange(5), 4)
    order = np.tile(np.arange(4), 5)
    subdomain = np.concatenate([4 * m + subdomain_for_order(np.arange(4), 4)
                                for m in range(5)])
    path = tmp_path / "tiny.npz"
    build_trajectory(coords, np.array([0, 50]), monomer, subdomain, order, path)
    with np.load(path) as z:
        assert z["x"].shape == (2, 20, 9)
        assert z["y"].shape == (2, 74)
        assert z["edge_ptr"].shape == (3,)
