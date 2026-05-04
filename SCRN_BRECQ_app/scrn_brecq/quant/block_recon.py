"""SCRN-BRECQ block reconstruction。

本文件参考 `BRECQ-main/quant/block_recon.py` 的优化流程，重新实现 SCRN 可用的
block reconstruction。它会把目标 block 内的权重量化器替换为 AdaRound，并用第七
部分缓存的输入输出数据优化 rounding 参数。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.adaptive_rounding import AdaRoundQuantizer
from SCRN_BRECQ_app.scrn_brecq.quant.data_utils import save_grad_data, save_inp_oup_data
from SCRN_BRECQ_app.scrn_brecq.quant.quant_block import BaseQuantBlock
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule, StraightThrough, lp_loss
from SCRN_BRECQ_app.scrn_brecq.quant.quant_model import QuantModel


ACTIVATION_DELTA_MIN = 1e-8


def block_reconstruction(
    model: QuantModel,
    block: BaseQuantBlock,
    cali_data: torch.Tensor,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = 0.01,
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
    """对一个 SCRN quant block 执行 BRECQ reconstruction。

    参数基本保持 BRECQ 原接口。`act_quant=False` 时优化 AdaRound `alpha`；
    `act_quant=True` 时只优化 block 内部 `QuantModule.act_quantizer.delta`。
    """
    _validate_reconstruction_args(block, cali_data, batch_size, iters, opt_mode, multi_gpu)
    quant_modules = _quant_modules_in_block(block)
    if not quant_modules:
        raise RuntimeError("Block reconstruction requires at least one QuantModule inside the block.")

    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    original_activation = block.activation_function
    if not include_act_func:
        block.activation_function = StraightThrough()

    try:
        if not act_quant:
            opt_params = _prepare_adaround_params(quant_modules)
            optimizer = torch.optim.Adam(opt_params)
            scheduler = None
            round_loss_mode = "relaxation"
        else:
            # 激活量化参数要先通过一次目标 block 前向初始化，再收集可学习 delta。
            cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, act_quant, batch_size)
            _initialize_activation_quantizers(block, cached_inps, batch_size)
            opt_params = _collect_activation_delta_params(quant_modules)
            optimizer = torch.optim.Adam(opt_params, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.0)
            round_loss_mode = "none"
        if not act_quant:
            cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, act_quant, batch_size)

        cached_grads = None
        if opt_mode != "mse":
            cached_grads = save_grad_data(model, block, cali_data, act_quant=act_quant, batch_size=batch_size)

        loss_func = BlockLossFunction(
            block,
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
            out_quant = block(cur_inp)
            err = loss_func(out_quant, cur_out, cur_grad)
            err.backward()
            _sync_parameter_gradients(opt_params, multi_gpu=multi_gpu)
            optimizer.step()
            if act_quant:
                _project_activation_delta_params_positive(opt_params)
            if scheduler is not None:
                scheduler.step()
    finally:
        if not act_quant:
            _set_adaround_soft_targets(quant_modules, enabled=False)
        if not include_act_func:
            block.activation_function = original_activation
        model.set_quant_state(False, False)
        block.set_quant_state(True, act_quant)
        _empty_cuda_cache_if_needed()


class BlockLossFunction:
    """block reconstruction 的损失函数。

    总损失 = 输出重构误差 + AdaRound rounding 正则。激活量化阶段不使用 rounding
    正则，只优化 activation scale。
    """

    def __init__(
        self,
        block: BaseQuantBlock,
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
        self.block = block
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
        """计算当前 batch 的 reconstruction loss。"""
        self.count += 1
        rec_loss = reconstruction_loss(pred, tgt, grad, rec_loss=self.rec_loss, p=self.p)
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            round_loss = pred.new_tensor(0.0)
            b = 0.0
        elif self.round_loss == "relaxation":
            round_loss = pred.new_tensor(0.0)
            for module in _quant_modules_in_block(self.block):
                round_loss = round_loss + _rounding_regularizer(module, self.weight, b)
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


class LinearTempDecay:
    """AdaRound rounding 正则的温度线性衰减器。"""

    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: float = 10,
        end_b: float = 2,
    ) -> None:
        if int(t_max) <= 0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        self.t_max = int(t_max)
        self.start_decay = float(rel_start_decay) * self.t_max
        self.start_b = float(start_b)
        self.end_b = float(end_b)

    def __call__(self, t: int) -> float:
        """返回第 t 次迭代使用的温度 b。"""
        if float(t) < self.start_decay:
            return self.start_b
        denom = max(float(self.t_max) - self.start_decay, 1e-12)
        rel_t = (float(t) - self.start_decay) / denom
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel_t)


def reconstruction_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    grad: torch.Tensor | None = None,
    *,
    rec_loss: str = "mse",
    p: float = 2.0,
) -> torch.Tensor:
    """计算 SCRN reconstruction 使用的输出误差。

    `mse` 复用 BRECQ 的 Lp loss；Fisher 模式对 batch 以外所有维度展平，兼容 Conv
    输出 `[N, C, H, W]`、Linear 输出 `[N, D]` 和 block 输出。
    """
    if rec_loss == "mse":
        return lp_loss(pred, tgt, p=p)
    if grad is None:
        raise ValueError(f"{rec_loss} reconstruction requires cached gradients.")

    diff = pred - tgt
    diff_flat = diff.reshape(diff.shape[0], -1)
    grad_flat = grad.reshape(grad.shape[0], -1)
    if rec_loss == "fisher_diag":
        return (diff_flat.pow(2) * grad_flat.pow(2)).sum(dim=1).mean()
    if rec_loss == "fisher_full":
        abs_diff = diff_flat.abs()
        abs_grad = grad_flat.abs()
        batch_dotprod = (abs_diff * abs_grad).sum(dim=1, keepdim=True)
        return (batch_dotprod * abs_diff * abs_grad).mean() / 100.0
    raise ValueError(f"Unsupported reconstruction loss function: {rec_loss}")


def _validate_reconstruction_args(
    target: nn.Module,
    cali_data: torch.Tensor,
    batch_size: int,
    iters: int,
    opt_mode: str,
    multi_gpu: bool,
) -> None:
    """检查 reconstruction 公共参数。"""
    if multi_gpu and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError("multi_gpu=True requires an initialized torch.distributed process group.")
    if not isinstance(target, (QuantModule, BaseQuantBlock)):
        raise TypeError(f"Expected QuantModule or BaseQuantBlock, got {type(target)!r}")
    if not isinstance(cali_data, torch.Tensor):
        raise TypeError(f"cali_data must be a torch.Tensor, got {type(cali_data)!r}")
    if int(cali_data.size(0)) <= 0:
        raise ValueError("cali_data must contain at least one sample.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if int(iters) <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    if opt_mode not in {"mse", "fisher_diag", "fisher_full"}:
        raise ValueError(f"Unsupported opt_mode: {opt_mode}")


def _quant_modules_in_block(block: BaseQuantBlock) -> list[QuantModule]:
    """按模型遍历顺序取出 block 内所有 QuantModule。"""
    return [module for module in block.modules() if isinstance(module, QuantModule)]


def _prepare_adaround_params(modules: list[QuantModule]) -> list[nn.Parameter]:
    """把权重量化器替换成 AdaRound，并返回待优化的 alpha 参数。"""
    opt_params: list[nn.Parameter] = []
    for module in modules:
        if isinstance(module.weight_quantizer, AdaRoundQuantizer):
            module.weight_quantizer.soft_targets = True
            if module.weight_quantizer.alpha is None:
                raise RuntimeError("AdaRoundQuantizer alpha is not initialized.")
            opt_params.append(module.weight_quantizer.alpha)
            continue

        _initialize_weight_quantizer(module)
        module.weight_quantizer = AdaRoundQuantizer(
            uaq=module.weight_quantizer,
            round_mode="learned_hard_sigmoid",
            weight_tensor=module.org_weight.detach(),
        )
        module.weight_quantizer.soft_targets = True
        if module.weight_quantizer.alpha is None:
            raise RuntimeError("AdaRoundQuantizer alpha is not initialized.")
        opt_params.append(module.weight_quantizer.alpha)
    return opt_params


def _initialize_weight_quantizer(module: QuantModule) -> None:
    """确保 UniformAffineQuantizer 已有 delta/zero_point，供 AdaRound 复制。"""
    quantizer = module.weight_quantizer
    if getattr(quantizer, "delta", None) is not None and getattr(quantizer, "zero_point", None) is not None:
        return
    with torch.no_grad():
        _ = quantizer(module.org_weight.detach())


def _initialize_activation_quantizers(target: nn.Module, cached_inps: torch.Tensor, batch_size: int) -> None:
    """通过一次局部前向初始化 activation quantizer 的可学习 delta。"""
    sample_size = min(int(batch_size), int(cached_inps.size(0)))
    if isinstance(target, (QuantModule, BaseQuantBlock)):
        target.set_quant_state(True, True)
    with torch.no_grad():
        _ = target(cached_inps[:sample_size])


def _collect_activation_delta_params(modules: list[QuantModule]) -> list[nn.Parameter]:
    """收集内部 QuantModule 的 activation scale 参数。"""
    opt_params: list[nn.Parameter] = []
    for module in modules:
        delta = module.act_quantizer.delta
        if isinstance(delta, nn.Parameter):
            opt_params.append(delta)
        elif delta is not None:
            raise RuntimeError(
                "Activation quantizer delta is not learnable. "
                "Construct QuantModel with act_quant_params['leaf_param']=True."
            )
    if not opt_params:
        raise RuntimeError(
            "No learnable activation delta found. "
            "Construct QuantModel with act_quant_params['leaf_param']=True and initialize activation quantization."
        )
    return opt_params


def _project_activation_delta_params_positive(
    params: list[nn.Parameter],
    eps: float = ACTIVATION_DELTA_MIN,
) -> None:
    """Project learnable activation scales back to the valid positive range."""
    eps_value = float(eps)
    if eps_value <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    with torch.no_grad():
        for param in params:
            param.clamp_(min=eps_value)


def _set_adaround_soft_targets(modules: list[QuantModule], *, enabled: bool) -> None:
    """批量切换 AdaRound soft/hard target。"""
    for module in modules:
        if isinstance(module.weight_quantizer, AdaRoundQuantizer):
            module.weight_quantizer.soft_targets = bool(enabled)


def _rounding_regularizer(module: QuantModule, weight: float, b: float) -> torch.Tensor:
    """计算单个 QuantModule 的 AdaRound rounding 正则项。"""
    if not hasattr(module.weight_quantizer, "get_soft_targets"):
        return module.org_weight.new_tensor(0.0)
    round_vals = module.weight_quantizer.get_soft_targets()
    return float(weight) * (1.0 - ((round_vals - 0.5).abs() * 2.0).pow(float(b))).sum()


def _sample_indices(cached_inps: torch.Tensor, sample_size: int) -> torch.Tensor:
    """从缓存样本中随机取一个 mini-batch。"""
    return torch.randperm(int(cached_inps.size(0)), device=cached_inps.device)[:sample_size]


def _sync_parameter_gradients(params: list[nn.Parameter], *, multi_gpu: bool) -> None:
    """分布式 reconstruction 时同步待优化参数梯度。"""
    if not multi_gpu:
        return
    world_size = dist.get_world_size()
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _empty_cuda_cache_if_needed() -> None:
    """CUDA 可用时释放缓存，CPU 环境不做处理。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
