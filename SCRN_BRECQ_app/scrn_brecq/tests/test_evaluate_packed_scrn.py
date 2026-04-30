import unittest

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn import checkpoint_like_from_manifest


class EvaluatePackedScrnTest(unittest.TestCase):
    def test_checkpoint_like_from_manifest_exposes_model_and_quant_config(self) -> None:
        manifest = {
            "source_checkpoint": "source.pth",
            "quant_checkpoint": "quant.pth",
            "final_quant_state": {"weight_quant": True, "act_quant": False},
            "checkpoint_metadata": {
                "checkpoint_stage": "final",
                "model_config": {"dim": 64},
                "quant_config": {"n_bits_w": 4, "act_quant": False},
            },
        }

        checkpoint = checkpoint_like_from_manifest(manifest)

        self.assertEqual(checkpoint["model_config"], {"dim": 64})
        self.assertEqual(checkpoint["quant_config"], {"n_bits_w": 4, "act_quant": False})
        self.assertEqual(checkpoint["source_checkpoint"], "source.pth")
        self.assertEqual(checkpoint["quant_checkpoint"], "quant.pth")
        self.assertEqual(checkpoint["final_quant_state"], {"weight_quant": True, "act_quant": False})


if __name__ == "__main__":
    unittest.main()
