import unittest
import torch
from lluna import (
    ExpertSpec,
    Lluna,
    agreement_graph,
    modality_views,
    cover_edges,
    masked_view,
    pcc_loss,
)
from lluna.pcc import edge_weights


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_graph(self):
        logits = torch.randn(20, 4, 2, requires_grad=True)
        graph = agreement_graph(logits, top_k=1)
        self.assertTrue(torch.equal(graph, graph.T))
        self.assertFalse(graph.requires_grad)
        self.assertTrue(torch.equal(graph.diag(), torch.ones(4)))

    def test_masking_residual_and_gradients(self):
        specs = [ExpertSpec("a", 5), ExpertSpec("b", 7)]
        model = Lluna(
            specs, torch.ones(2, 2), output_dim=2, hidden_dim=8, dropout=0
        ).eval()
        x = {"a": torch.randn(3, 5), "b": torch.randn(3, 7)}
        deps = {"a": ("A",), "b": ("B",)}
        o = model(masked_view(x, deps, {"A"}))
        x["b"] = torch.randn(3, 7) * 10000
        other = model(masked_view(x, deps, {"A"}))
        self.assertTrue(torch.equal(o["logits"], other["logits"]))
        torch.testing.assert_close(
            o["logits"],
            o["private_logits"] + o["residual_scale"] * o["common_residual_logits"],
        )
        self.assertEqual(o["edges"][:, 1, :].abs().sum().item(), 0)
        self.assertEqual(o["edges"][:, :, 1].abs().sum().item(), 0)
        o["logits"].sum().backward()
        self.assertIsNotNone(model.common_aligner.net[1].weight.grad)

    def test_pcc_direction(self):
        p, c = frozenset("AB"), frozenset("A")
        teacher = torch.randn(3, 8, requires_grad=True)
        student = torch.randn(3, 8, requires_grad=True)
        pcc_loss({p: teacher, c: student}, [(p, c)]).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_lattice_weights(self):
        views = modality_views("TAV")
        edges = cover_edges(views)
        w = edge_weights(edges)
        self.assertEqual(len(views), 7)
        self.assertEqual(len(edges), 9)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(sum(v for (p, c), v in w.items() if len(c) == 1), 0.5)


if __name__ == "__main__":
    unittest.main()
