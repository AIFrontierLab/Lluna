import json
from pathlib import Path
import tempfile
import unittest
import torch
from example import make_data
from lluna.data import FeatureDataset
from lluna.runner import train, test


class PipelineTests(unittest.TestCase):
    def test_classification_and_regression(self):
        torch.set_num_threads(2)
        for task, sites in [
            ("classification", False),
            ("regression", False),
            ("candidate", False),
            ("classification", True),
        ]:
            with self.subTest(
                task=task, sites=sites
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                make_data(root / "data", task, sites)
                # Training must work without any test file.
                (root / "data/test.pt").rename(root / "held_out.pt")
                data = FeatureDataset(root / "data")
                config = json.loads(
                    (Path(__file__).parents[1] / "configs/default.json").read_text()
                )
                config["heads"]["epochs"] = 1
                config["model"].update(hidden_dim=8, common_dim=16)
                config["training"].update(
                    epochs=2,
                    patience=2,
                    selection_metric="mae" if task == "regression" else "accuracy",
                    view_sampling="chain" if sites or task == "candidate" else "full",
                )
                train(data, config, root / "run", torch.device("cpu"))
                self.assertNotIn("test", data.splits)
                self.assertFalse((root / "run/predictions.pt").exists())
                with self.assertRaises(FileExistsError):
                    train(data, config, root / "run", torch.device("cpu"))
                (root / "held_out.pt").rename(root / "data/test.pt")
                test(
                    data,
                    root / "run/checkpoint.pt",
                    root / "evaluation",
                    torch.device("cpu"),
                )
                results = json.loads((root / "evaluation/metrics.json").read_text())
                self.assertEqual(len(results["views"]), 15 if sites else 7)
                self.assertTrue((root / "evaluation/metrics.csv").exists())

    def test_partition_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            make_data(root)
            validation = torch.load(root / "validation.pt", weights_only=True)
            validation["ids"][0] = "example_0"
            torch.save(validation, root / "validation.pt")
            with self.assertRaisesRegex(ValueError, "overlap"):
                FeatureDataset(root)


if __name__ == "__main__":
    unittest.main()
