import json
from pathlib import Path
import tempfile
import unittest
import torch
from example import make_data
from lluna.data import FeatureDataset
from lluna.graph import agreement_graph
from lluna.model import Lluna, ExpertSpec
from lluna.pcc import pcc_loss, modality_views
from lluna.sampling import ViewSampler


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_bad_controls(self):
        for kwargs in ({"temperature": float("nan")}, {"self_loop": float("inf")}):
            with self.assertRaises(ValueError):
                agreement_graph(torch.randn(4, 2, 2), **kwargs)
        with self.assertRaises(ValueError):
            Lluna(
                [ExpertSpec("a", 3), ExpertSpec("b", 3)],
                torch.eye(2),
                output_dim=2,
                hidden_dim=4,
                common_dim=0,
            )

    def test_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            make_data(root)
            v = torch.load(root / "validation.pt", weights_only=True)
            v["group_ids"][0] = "group_0"
            torch.save(v, root / "validation.pt")
            with self.assertRaisesRegex(ValueError, "Groups overlap"):
                FeatureDataset(root)

    def test_site_mask_invariance(self):
        specs = [ExpertSpec("signal", 5), ExpertSpec("context", 3)]
        model = Lluna(
            specs,
            torch.ones(2, 2),
            output_dim=2,
            hidden_dim=8,
            dropout=0,
            site_experts=["signal"],
            site_count=4,
            site_projection_dim=4,
        ).eval()
        batch = {
            "features": {"signal": torch.randn(2, 4, 5), "context": torch.randn(2, 3)},
            "available": {
                k: torch.ones(2, dtype=torch.bool) for k in ("signal", "context")
            },
            "site_mask": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
        }
        before = model(batch)["logits"]
        batch["features"]["signal"][~batch["site_mask"]] = 10000
        torch.testing.assert_close(before, model(batch)["logits"], atol=0, rtol=0)

    def test_candidate_padding_and_equal_row_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            make_data(root, "candidate")
            data = FeatureDataset(root)
            b, y = data.batch("train", [0, 1], data.views[0], torch.device("cpu"))
            model = data.model(torch.eye(9), dict(hidden_dim=8, dropout=0))
            o = data.predictions(model(b), b)
            mask = b["candidate_mask"]
            self.assertEqual(o["logits"].shape, (2, 4))
            self.assertEqual(o["logits"][0, 3], torch.finfo(torch.float32).min)
            self.assertTrue(mask[torch.arange(2), o["logits"].argmax(-1)].all())
        p, c = frozenset("AB"), frozenset("A")
        t = torch.tensor([[[1.0, 0.0]] * 4] * 2, requires_grad=True)
        s = torch.tensor([[[0.0, 1.0]] * 4, [[1.0, 0.0]] * 4], requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        loss = pcc_loss({p: t, c: s}, [(p, c)], candidate_mask=mask)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertEqual(s.grad[~mask].abs().sum().item(), 0)

    def test_chain_expectation(self):
        sampler = ViewSampler(modality_views("TAV"), "chain")
        # Exact expectation across all uniform chains for arbitrary scalar terms.
        task_sum = {v: 0.0 for v in sampler.views}
        edge_sum = {e: 0.0 for e in sampler.edges}
        for chain in sampler.chains:
            for v in chain:
                task_sum[v] += (
                    sampler.task[v] / sampler.view_probability[v] / len(sampler.chains)
                )
            for e in zip(chain, chain[1:]):
                edge_sum[e] += (
                    sampler.pcc[e] / sampler.edge_probability[e] / len(sampler.chains)
                )
        for v in sampler.views:
            self.assertAlmostEqual(task_sum[v], sampler.task[v])
        for e in sampler.edges:
            self.assertAlmostEqual(edge_sum[e], sampler.pcc[e])
        self.assertEqual(sampler.sample(8), sampler.sample(8))


if __name__ == "__main__":
    unittest.main()
