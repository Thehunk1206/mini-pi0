import unittest

import torch

from mini_pi0.eval.runner import _select_checkpoint_model_state


class EvalWeightSourceTests(unittest.TestCase):
    def test_select_checkpoint_model_state_uses_canonical_model(self):
        model = {"w": torch.tensor([1.0])}

        selected = _select_checkpoint_model_state({"model": model}, "model")

        self.assertIs(selected, model)

    def test_select_checkpoint_model_state_uses_raw_when_available(self):
        raw = {"w": torch.tensor([1.0])}
        ema = {"w": torch.tensor([2.0])}

        selected = _select_checkpoint_model_state(
            {"model": ema, "model_raw": raw, "model_weight_source": "ema"},
            "raw",
        )

        self.assertIs(selected, raw)

    def test_select_checkpoint_model_state_uses_ema_shadow(self):
        raw = {"w": torch.tensor([1.0])}
        ema = {"w": torch.tensor([2.0])}

        selected = _select_checkpoint_model_state(
            {"model": raw, "ema": {"decay": 0.999, "shadow": ema}},
            "ema",
        )

        self.assertIs(selected, ema)

    def test_select_checkpoint_model_state_rejects_missing_raw(self):
        with self.assertRaises(ValueError):
            _select_checkpoint_model_state({"model": {"w": torch.tensor([1.0])}}, "raw")


if __name__ == "__main__":
    unittest.main()
