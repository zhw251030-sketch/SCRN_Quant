"""SCRN-BRECQ layer reconstruction。

本文件实现单个 `QuantModule` 的 reconstruction。它和 block reconstruction 使用相同
的 AdaRound 正则、输入输出缓存和 SCRN 回归任务适配，只是重构单位从 block 变成
一个 Conv/Linear 包装层。
"""

from __future__ import annotations

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.adaptive_rounding import AdaRoundQuantizer
from SCRN_BRECQ_app.scrn_brecq.quant.block_recon import (
    LinearTempDecay,
    _collect_activation_delta_params,
    _empty_cuda_cache_if_needed,
    _initialize_activation_quantizers,
    _prepare_adaround_params,
    _project_activation_delta_params_positive,
    _rounding_regularizer,
    _sample_indices,
    _sync_parameter_gradients,
    _validate_reconstruction_args,
    reconstruction_loss,
)
from SCRN_BRECQ_app.scrn_brecq.quant.data_utils import save_grad_data, save_inp_oup_data
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule, StraightThrough
from SCRN_BRECQ_app.scrn_brecq.quant.quant_model import QuantModel


def layer_reconstruction(
    model: QuantModel,
    layer: QuantModule,
    cali_data: torch.Tensor,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = 0.001,
    opt_mode: str = "mse",
    asym: bool = False,
    include_act_func: bool = True,
    b_range: tuple[float, float] = (20, 2),
    warmup: float = 0.0,
    act_quant: bool = False,
    lr: float = 4e-5,
    p: float = 2.0,
    multi_gpu: bool = False,
    log_enabled: bool = True,
) -> None:
    """对单个 QuantModule 执行 BRECQ reconstruction。

    `act_quant=False` 时执行 AdaRound 权重量化；`act_quant=True` 时优化该层的
    activation quantizer scale。
    """
    if not isinstance(layer, QuantModule):
        raise TypeError(f"layer_reconstruction expects QuantModule, got {type(layer)!r}")
    _validate_reconstruction_args(layer, cali_data, batch_size, iters, opt_mode, multi_gpu)

    model.set_quant_state(False, False)
    layer.set_quant_state(True, act_quant)
    original_activation = layer.activation_function
    if not include_act_func:
        layer.activation_function = StraightThrough()

    try:
        if not act_quant:
            opt_params = _prepare_adaround_params([layer])
            optimizer = torch.optim.Adam(opt_params)
            scheduler = None
            round_loss_mode = "relaxation"
        else:
            cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)
            _initialize_activation_quantizers(layer, cached_inps, batch_size)
            opt_params = _collect_activation_delta_params([layer])
            optimizer = torch.optim.Adam(opt_params, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.0)
            round_loss_mode = "none"
        if not act_quant:
            cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)

        cached_grads = None
        if opt_mode != "mse":
            cached_grads = save_grad_data(model, layer, cali_data, act_quant=act_quant, batch_size=batch_size)

        loss_func = LayerLossFunction(
            layer,
            round_loss=round_loss_mode,
            weight=weight,
            max_count=iters,
            rec_loss=opt_mode,
            b_range=b_range,
            decay_start=0.0,
            warmup=warmup,
            p=p,
            log_enabled=log_enabled,
        )

        sample_size = min(int(batch_size), int(cached_inps.size(0)))
        for _ in range(int(iters)):
            idx = _sample_indices(cached_inps, sample_size)
            cur_inp = cached_inps[idx]
            cur_out = cached_outs[idx]
            cur_grad = cached_grads[idx] if cached_grads is not None else None

            optimizer.zero_grad()
            out_quant = layer(cur_inp)
            err = loss_func(out_quant, cur_out, cur_grad)
            err.backward()
            _sync_parameter_gradients(opt_params, multi_gpu=multi_gpu)
            optimizer.step()
            if act_quant:
                _project_activation_delta_params_positive(opt_params)
            if scheduler is not None:
                scheduler.step()
    finally:
        if isinstance(layer.weight_quantizer, AdaRoundQuantizer):
            layer.weight_quantizer.soft_targets = False
        if not include_act_func:
            layer.activation_function = original_activation
        model.set_quant_state(False, False)
        layer.set_quant_state(True, act_quant)
        _empty_cuda_cache_if_needed()


class LayerLossFunction:
    """单层 reconstruction 损失函数。"""

    def __init__(
        self,
        layer: QuantModule,
        round_loss: str = "relaxation",
        weight: float = 1.0,
        rec_loss: str = "mse",
        max_count: int = 2000,
        b_range: tuple[float, float] = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
        log_enabled: bool = True,
    ) -> None:
        self.layer = layer
        self.round_loss = str(round_loss)
        self.weight = float(weight)
        self.rec_loss = str(rec_loss)
        self.loss_start = int(max_count) * float(warmup)
        self.p = float(p)
        self.log_enabled = bool(log_enabled)
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=float(warmup) + (1.0 - float(warmup)) * float(decay_start),
            start_b=float(b_range[0]),
            end_b=float(b_range[1]),
        )
        self.count = 0

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor, grad: torch.Tensor | None = None) -> torch.Tensor:
        """计算当前 batch 的 layer reconstruction loss。"""
        self.count += 1
        rec_loss = reconstruction_loss(pred, tgt, grad, rec_loss=self.rec_loss, p=self.p)
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            round_loss = pred.new_tensor(0.0)
            b = 0.0
        elif self.round_loss == "relaxation":
            round_loss = _rounding_regularizer(self.layer, self.weight, b)
        else:
            raise NotImplementedError(f"Unsupported round loss: {self.round_loss}")

        total_loss = rec_loss + round_loss
        if self.log_enabled and self.count % 500 == 0:
            print(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss.detach()),
                    float(rec_loss.detach()),
                    float(round_loss.detach()),
                    float(b),
                    self.count,
                )
            )
        return total_loss


__all__ = ["LayerLossFunction", "layer_reconstruction"]
