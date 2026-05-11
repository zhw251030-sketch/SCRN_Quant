import unittest

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import (
    apply_activation_sensitivity_mode,
    select_activation_quantizers,
)


def _quant_module(module: nn.Module) -> QuantModule:
    return QuantModule(
        module,
        weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
        act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
    )


def _model() -> nn.Sequential:
    model = nn.Sequential(
        nn.ModuleDict(
            {
                "stage1": nn.Sequential(
                    nn.ModuleDict(
                        {
                            "cnn_branch": nn.Sequential(_quant_module(nn.Conv2d(1, 2, kernel_size=1))),
                        }
                    )
                ),
                "stage4": nn.Sequential(
                    nn.ModuleDict(
                        {
                            "trans_branch": nn.ModuleDict(
                                {
                                    "attn": nn.ModuleDict(
                                        {
                                            "qkv": _quant_module(nn.Linear(2, 2)),
                                            "proj": _quant_module(nn.Linear(2, 2)),
                                        }
                                    )
                                }
                            )
                        }
                    )
                ),
            }
        )
    )
    return model


class ActivationSensitivityTest(unittest.TestCase):
    def test_selector_filters_by_structure_and_excludes_output_by_default(self) -> None:
        rows = select_activation_quantizers(_model(), branch="transformer", include_output_quantizer=False)

        self.assertEqual([row["role"] for row in rows], ["attention_qkv"])
        self.assertEqual(rows[0]["branch"], "transformer")
        self.assertEqual(rows[0]["module_type"], "Linear")

    def test_selector_can_include_output_quantizer(self) -> None:
        rows = select_activation_quantizers(_model(), branch="transformer", include_output_quantizer=True)

        self.assertEqual([row["role"] for row in rows], ["attention_qkv", "attention_proj"])

    def test_selector_filters_by_index_name_stage_role_and_module_type(self) -> None:
        model = _model()

        by_index = select_activation_quantizers(model, index=0)
        by_name = select_activation_quantizers(model, name_contains="attn.qkv")
        by_stage_role = select_activation_quantizers(model, stage="stage4", role="attention_qkv")
        by_type = select_activation_quantizers(model, module_type="Conv2d")

        self.assertEqual(by_index[0]["name"], "0.stage1.0.cnn_branch.0")
        self.assertEqual(by_name[0]["role"], "attention_qkv")
        self.assertEqual(by_stage_role[0]["name"], by_name[0]["name"])
        self.assertEqual(by_type[0]["branch"], "cnn")

    def test_selector_accepts_plural_or_filters(self) -> None:
        model = _model()

        by_roles = select_activation_quantizers(
            model,
            roles=("attention_qkv", "attention_proj"),
            include_output_quantizer=True,
        )
        by_stages = select_activation_quantizers(model, stages=("stage1", "stage4"), include_output_quantizer=True)
        by_branches = select_activation_quantizers(model, branches=("cnn", "transformer"), include_output_quantizer=True)
        by_types = select_activation_quantizers(model, module_types=("Conv2d", "Linear"), include_output_quantizer=True)

        self.assertEqual([row["role"] for row in by_roles], ["attention_qkv", "attention_proj"])
        self.assertEqual([row["stage"] for row in by_stages], ["stage1", "stage4", "stage4"])
        self.assertEqual([row["branch"] for row in by_branches], ["cnn", "transformer", "transformer"])
        self.assertEqual([row["module_type"] for row in by_types], ["Conv2d", "Linear", "Linear"])

    def test_modes_set_disable_act_quant_and_restore_original_state(self) -> None:
        model = _model()
        modules = [module for module in model.modules() if isinstance(module, QuantModule)]
        modules[0].disable_act_quant = True

        with apply_activation_sensitivity_mode(model, mode="all_on") as selected:
            self.assertEqual(len(selected), 2)
            self.assertEqual([module.disable_act_quant for module in modules], [False, False, True])
        self.assertEqual([module.disable_act_quant for module in modules], [True, False, False])

        with apply_activation_sensitivity_mode(model, mode="all_off"):
            self.assertEqual([module.disable_act_quant for module in modules], [True, True, True])
        self.assertEqual([module.disable_act_quant for module in modules], [True, False, False])

    def test_disable_one_and_enable_one_modes(self) -> None:
        model = _model()
        modules = [module for module in model.modules() if isinstance(module, QuantModule)]

        with apply_activation_sensitivity_mode(model, mode="disable_one", index=1) as selected:
            self.assertEqual([row["index"] for row in selected], [1])
            self.assertEqual([module.disable_act_quant for module in modules], [False, True, True])

        with apply_activation_sensitivity_mode(model, mode="enable_one", index=1) as selected:
            self.assertEqual([row["index"] for row in selected], [1])
            self.assertEqual([module.disable_act_quant for module in modules], [True, False, True])

    def test_group_modes_select_matching_quantizers(self) -> None:
        model = _model()
        modules = [module for module in model.modules() if isinstance(module, QuantModule)]

        with apply_activation_sensitivity_mode(model, mode="disable_group", branch="transformer") as selected:
            self.assertEqual([row["role"] for row in selected], ["attention_qkv"])
            self.assertEqual([module.disable_act_quant for module in modules], [False, True, True])

        with apply_activation_sensitivity_mode(model, mode="enable_group", branch="transformer") as selected:
            self.assertEqual([row["role"] for row in selected], ["attention_qkv"])
            self.assertEqual([module.disable_act_quant for module in modules], [True, False, True])

        with apply_activation_sensitivity_mode(
            model,
            mode="disable_group",
            roles=("attention_qkv", "attention_proj"),
            include_output_quantizer=True,
        ) as selected:
            self.assertEqual([row["role"] for row in selected], ["attention_qkv", "attention_proj"])
            self.assertEqual([module.disable_act_quant for module in modules], [False, True, True])


if __name__ == "__main__":
    unittest.main()
