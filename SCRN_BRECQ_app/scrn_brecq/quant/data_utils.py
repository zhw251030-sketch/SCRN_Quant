"""BRECQ reconstruction 数据缓存工具。

Layer reconstruction 和 block reconstruction 在优化 AdaRound 参数前，需要先固定一批
calibration 输入，并缓存目标层/块在 FP32 路径下的输入和输出。本文件参考
`BRECQ-main/quant/data_utils.py` 的 hook 思路重新实现，不导入原 BRECQ 源码。

SCRN 是图像恢复模型，输出是 `[N, 1, H, W]` 回归结果，不是分类 logits。因此梯度
缓存不使用原 BRECQ 的 KL loss，而是用量化输出和 FP32 输出之间的 MSE loss。
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F

from SCRN_BRECQ_app.scrn_brecq.quant.quant_block import BaseQuantBlock
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.quant_model import QuantModel


QuantTarget = Union[QuantModule, BaseQuantBlock]


def save_inp_oup_data(
    model: QuantModel,
    layer: QuantTarget,
    cali_data: torch.Tensor,
    asym: bool = False,
    act_quant: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缓存目标层/块在 calibration data 上的输入和输出。

    参数:
        model: 已经用 `QuantModel` 包装过的 SCRN。
        layer: 需要重构的 `QuantModule` 或 `BaseQuantBlock`。
        cali_data: SCRN calibration 输入，形状通常是 `[N, 1, H, W]`。
        asym: AdaRound 的非对称重构模式。为 True 时，输入来自量化前序网络，
            输出仍来自 FP32 目标层/块。
        act_quant: 采集量化输入时是否打开激活量化。
        batch_size: 分批前向的 batch 大小；最后不足一批的数据也会保留。
        keep_gpu: 是否把最终缓存放回模型所在设备，False 时返回 CPU tensor。

    返回:
        `(cached_inps, cached_outs)`，第一维都等于 `cali_data.size(0)`。
    """
    _validate_cache_inputs(layer, cali_data, batch_size)
    device = _model_device(model)
    getter = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    _empty_cuda_cache_if_needed()

    for batch in _iter_cali_batches(cali_data, batch_size):
        cur_inp, cur_out = getter(batch)
        cached_batches.append((cur_inp.detach().cpu(), cur_out.detach().cpu()))

    cached_inps = torch.cat([item[0] for item in cached_batches], dim=0).contiguous()
    cached_outs = torch.cat([item[1] for item in cached_batches], dim=0).contiguous()
    _empty_cuda_cache_if_needed()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs


def save_grad_data(
    model: QuantModel,
    layer: QuantTarget,
    cali_data: torch.Tensor,
    damping: float = 1.0,
    act_quant: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
) -> torch.Tensor:
    """缓存目标层/块输出处的重构梯度。

    原 BRECQ 面向分类模型，用 FP32 logits 和量化 logits 的 KL loss 计算 Fisher 近似。
    SCRN 输出是单通道恢复图像，分类 KL 不适用；这里使用 `MSE(out_q, out_fp)` 对目标
    层/块输出求梯度，并返回 `abs(grad) + damping`，供后续 Fisher 类重构损失使用。
    """
    _validate_cache_inputs(layer, cali_data, batch_size)
    if float(damping) < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")

    device = _model_device(model)
    getter = GetLayerGrad(model, layer, device=device, act_quant=act_quant)
    cached_batches: list[torch.Tensor] = []
    _empty_cuda_cache_if_needed()

    for batch in _iter_cali_batches(cali_data, batch_size):
        cur_grad = getter(batch)
        cached_batches.append(cur_grad.detach().cpu())

    cached_grads = torch.cat(cached_batches, dim=0).contiguous()
    cached_grads = cached_grads.abs() + float(damping)
    _empty_cuda_cache_if_needed()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """用于在 hook 保存目标层/块数据后主动截断前向传播。"""


class DataSaverHook:
    """保存 forward hook 看到的输入和输出。

    BRECQ 只需要目标层/块的局部输入输出。`stop_forward=True` 时，hook 保存数据后抛出
    `StopForwardException`，避免继续执行目标层/块之后的网络，节省显存和时间。
    """

    def __init__(self, store_input: bool = False, store_output: bool = False, stop_forward: bool = False) -> None:
        self.store_input = bool(store_input)
        self.store_output = bool(store_output)
        self.stop_forward = bool(stop_forward)
        self.input_store: tuple[torch.Tensor, ...] | torch.Tensor | None = None
        self.output_store: torch.Tensor | None = None

    def __call__(self, module, input_batch, output_batch) -> None:
        """forward hook 回调，保存需要的 tensor。"""
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            if not isinstance(output_batch, torch.Tensor):
                raise TypeError(f"Expected tensor output from hooked module, got {type(output_batch)!r}")
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

    def reset(self) -> None:
        """清空上一次 batch 缓存，避免异常情况下读到旧数据。"""
        self.input_store = None
        self.output_store = None


class GetLayerInpOut:
    """运行一次模型前向，并取出目标层/块的输入输出。"""

    def __init__(
        self,
        model: QuantModel,
        layer: QuantTarget,
        device: torch.device,
        asym: bool = False,
        act_quant: bool = False,
    ) -> None:
        self.model = model
        self.layer = layer
        self.asym = bool(asym)
        self.device = device
        self.act_quant = bool(act_quant)
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回当前 batch 的目标输入和 FP32 目标输出。"""
        was_training = self.model.training
        self.model.eval()
        self.model.set_quant_state(False, False)
        self.data_saver.reset()
        handle = self.layer.register_forward_hook(self.data_saver)

        try:
            with torch.no_grad():
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass

                if self.asym:
                    # asym 模式只重算输入：前序网络打开量化，目标输出仍保留第一次 FP32 输出。
                    self.data_saver.store_output = False
                    self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                    try:
                        _ = self.model(model_input.to(self.device))
                    except StopForwardException:
                        pass
                    self.data_saver.store_output = True
        finally:
            handle.remove()

        input_tensor = _first_tensor_from_hook_input(self.data_saver.input_store)
        output_tensor = self.data_saver.output_store
        if output_tensor is None:
            raise RuntimeError("Failed to capture target layer/block output.")

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train(was_training)
        return input_tensor.detach(), output_tensor.detach()


class GradSaverHook:
    """保存 backward hook 看到的目标层/块输出梯度。"""

    def __init__(self, store_grad: bool = True) -> None:
        self.store_grad = bool(store_grad)
        self.grad_out: torch.Tensor | None = None

    def __call__(self, module, grad_input, grad_output) -> None:
        """backward hook 回调，保存输出端梯度。"""
        if self.store_grad:
            if not grad_output or grad_output[0] is None:
                raise RuntimeError("Backward hook did not receive output gradient.")
            self.grad_out = grad_output[0]

    def reset(self) -> None:
        """清空上一次 batch 缓存。"""
        self.grad_out = None


class GetLayerGrad:
    """运行一次 FP32/量化模型对比，并取目标层/块输出梯度。"""

    def __init__(
        self,
        model: QuantModel,
        layer: QuantTarget,
        device: torch.device,
        act_quant: bool = False,
    ) -> None:
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = bool(act_quant)
        self.grad_saver = GradSaverHook(store_grad=True)

    def __call__(self, model_input: torch.Tensor) -> torch.Tensor:
        """返回当前 batch 在目标层/块输出处的 MSE 重构梯度。"""
        was_training = self.model.training
        self.model.eval()
        self.grad_saver.reset()
        handle = self.layer.register_full_backward_hook(self.grad_saver)

        try:
            # 让 batch 输入参与计算图，避免 PyTorch 在 full backward hook 中退化到
            # “仅输出需要梯度”的兼容路径；我们只读取目标层/块输出梯度。
            inputs = model_input.to(self.device).detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs).detach()

            with torch.enable_grad():
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.mse_loss(out_q, out_fp)
                loss.backward()
        finally:
            handle.remove()

        grad_out = self.grad_saver.grad_out
        if grad_out is None:
            raise RuntimeError("Failed to capture target layer/block output gradient.")

        self.model.zero_grad(set_to_none=True)
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train(was_training)
        return grad_out.detach()


def quantize_model_till(model: QuantModel, layer: QuantTarget, act_quant: bool = False) -> None:
    """把模型中目标层/块之前的量化模块打开，并在目标处停止。

    `asym=True` 和 Fisher 梯度缓存需要模拟“前序网络已量化，当前目标待重构”的状态。
    如果目标是 `BaseQuantBlock`，则整个 block 作为停止点；如果目标是 block 内部的
    `QuantModule`，则只逐层打开到该 module，避免提前打开同一 block 后面的层。
    """
    _validate_quant_target(layer)
    model.set_quant_state(False, False)
    for _, module in model.named_modules():
        if isinstance(module, BaseQuantBlock):
            if module is layer:
                module.set_quant_state(True, act_quant)
                break
            if _module_contains(module, layer):
                continue
            module.set_quant_state(True, act_quant)
        elif isinstance(module, QuantModule):
            module.set_quant_state(True, act_quant)
            if module is layer:
                break


def _iter_cali_batches(cali_data: torch.Tensor, batch_size: int):
    """按 batch_size 切分 calibration tensor，保留最后不足一批的数据。"""
    for start in range(0, int(cali_data.size(0)), int(batch_size)):
        yield cali_data[start : start + int(batch_size)]


def _first_tensor_from_hook_input(input_store) -> torch.Tensor:
    """从 forward hook 的输入对象中取出第一个 tensor。"""
    if isinstance(input_store, torch.Tensor):
        return input_store
    if isinstance(input_store, (tuple, list)) and input_store and isinstance(input_store[0], torch.Tensor):
        return input_store[0]
    raise RuntimeError(f"Failed to capture tensor input from hook, got {type(input_store)!r}")


def _module_contains(parent: torch.nn.Module, target: torch.nn.Module) -> bool:
    """判断 target 是否是 parent 的子模块，不把 parent 自身算作包含。"""
    return any(child is target for child in parent.modules() if child is not parent)


def _model_device(model: QuantModel) -> torch.device:
    """读取模型参数所在设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("QuantModel has no parameters, cannot infer device.") from exc


def _validate_cache_inputs(layer: QuantTarget, cali_data: torch.Tensor, batch_size: int) -> None:
    """检查缓存工具的公共输入，尽早暴露调用错误。"""
    _validate_quant_target(layer)
    if not isinstance(cali_data, torch.Tensor):
        raise TypeError(f"cali_data must be a torch.Tensor, got {type(cali_data)!r}")
    if cali_data.ndim < 2:
        raise ValueError(f"cali_data must contain a batch dimension, got shape {tuple(cali_data.shape)}")
    if int(cali_data.size(0)) <= 0:
        raise ValueError("cali_data must contain at least one sample.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _validate_quant_target(layer: QuantTarget) -> None:
    """检查目标是否是可重构的量化层或量化 block。"""
    if not isinstance(layer, (QuantModule, BaseQuantBlock)):
        raise TypeError(f"Expected QuantModule or BaseQuantBlock, got {type(layer)!r}")


def _empty_cuda_cache_if_needed() -> None:
    """在 CUDA 可用时释放 PyTorch 缓存，CPU 测试时不做任何事。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
