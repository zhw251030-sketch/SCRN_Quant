"""SCRN 独立复现训练入口。

默认配置参考公开 SCRN 源码：Adam、MSELoss(sum)、80 epoch、MultiStepLR(20/40/60)。
训练输入由 clean patch 在线退化得到，目标为 clean patch。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from SCRN_BRECQ_app.scrn_repro.data import SCRNPatchDataset, build_train_loader
from SCRN_BRECQ_app.scrn_repro.model import SCRN
from SCRN_BRECQ_app.scrn_repro.training import (
    append_csv_row,
    append_jsonl,
    collect_environment,
    create_run_dir,
    save_checkpoint,
    write_json,
    write_summary,
)
from SCRN_BRECQ_app.scrn_repro.utils import set_random_seed


DEFAULT_DATASET_DIR = "SCRN_BRECQ_app/scrn_repro/datasets/scrn_train_patches"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_repro/runs/train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the independently reproduced SCRN model.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="训练 clean patch .npy 目录")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="训练 run 输出根目录")
    parser.add_argument("--run-name", default="scrn_train", help="run 名称，会和时间戳组合成目录名")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpus", default="", help="选择 GPU，例如 `0` 或 `0,1`；多卡需配合 torchrun")
    parser.add_argument("--distributed", action="store_true", help="启用 torch.distributed DDP 训练")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--milestones", default="20,40,60", help="逗号分隔的学习率下降 epoch")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--stage-depths", default="1,1,1,1,1")
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--input-resolution", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.gpus:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)

    set_random_seed(args.seed)
    distributed_state = _init_distributed(args)
    rank = distributed_state["rank"]
    world_size = distributed_state["world_size"]
    is_main = rank == 0
    device = _select_device(args, distributed_state)

    run_dir = None
    if is_main:
        run_dir = create_run_dir(args.run_root, run_name=args.run_name)
        _write_initial_config(run_dir, args, device, rank, world_size)
    run_dir = _share_run_dir(run_dir, device, args.distributed)

    try:
        final_summary = train(args, run_dir, device, rank, world_size, is_main)
        if is_main:
            write_summary(
                run_dir / "summary.md",
                title="SCRN Training Run",
                sections={
                    "Final Metrics": final_summary,
                    "Artifacts": {
                        "latest": run_dir / "checkpoints" / "latest.pth",
                        "best": run_dir / "checkpoints" / "best.pth",
                    },
                },
            )
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


def train(args: argparse.Namespace, run_dir: Path, device: torch.device, rank: int, world_size: int, is_main: bool) -> dict[str, Any]:
    dataset = SCRNPatchDataset(args.dataset_dir, max_samples=args.max_train_samples, seed=args.seed)
    loader, sampler = build_train_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
    )

    model = SCRN(
        dim=args.dim,
        stage_depths=_parse_int_tuple(args.stage_depths, expected_len=5),
        head_dim=args.head_dim,
        window_size=args.window_size,
        drop_path_rate=args.drop_path_rate,
        input_resolution=args.input_resolution,
    ).to(device)
    if args.distributed:
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)

    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=_parse_int_tuple(args.milestones), gamma=args.gamma)

    best_loss = float("inf")
    global_step = 0
    metrics_path = run_dir / "metrics.csv"
    metrics_jsonl_path = run_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)

        start_time = time.time()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            loss_sum += float(loss.detach().item())
            sample_count += batch_size
            global_step += batch_size * world_size

        if args.distributed:
            totals = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_sum = float(totals[0].item())
            sample_count = int(totals[1].item())

        epoch_loss = loss_sum / max(sample_count, 1)
        scheduler.step()
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "lr": current_lr,
            "samples": sample_count,
            "global_step": global_step,
            "epoch_seconds": elapsed,
        }
        if is_main:
            append_csv_row(metrics_path, row, fieldnames=row.keys())
            append_jsonl(metrics_jsonl_path, row)
            checkpoint_payload = _checkpoint_payload(args, model, optimizer, scheduler, epoch, epoch_loss)
            save_checkpoint(run_dir / "checkpoints" / "latest.pth", checkpoint_payload)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(run_dir / "checkpoints" / "best.pth", checkpoint_payload)
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pth", checkpoint_payload)
            print(f"epoch={epoch} loss={epoch_loss:.6f} lr={current_lr:.6g} seconds={elapsed:.2f}", flush=True)

    return {"best_loss": best_loss, "last_loss": epoch_loss, "epochs": args.epochs}


def _checkpoint_payload(args, model, optimizer, scheduler, epoch: int, loss: float) -> dict[str, Any]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "dim": args.dim,
            "stage_depths": _parse_int_tuple(args.stage_depths, expected_len=5),
            "head_dim": args.head_dim,
            "window_size": args.window_size,
            "drop_path_rate": args.drop_path_rate,
            "input_resolution": args.input_resolution,
        },
        "args": vars(args),
    }


def _write_initial_config(run_dir: Path, args: argparse.Namespace, device: torch.device, rank: int, world_size: int) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "device": str(device),
        "rank": rank,
        "world_size": world_size,
        "environment": collect_environment(),
    }
    write_json(run_dir / "config.json", payload)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.gpus and "," in args.gpus and not args.distributed:
        raise ValueError("Multiple GPUs require --distributed and torchrun. Example: torchrun --nproc_per_node=2 ... --distributed")


def _init_distributed(args: argparse.Namespace) -> dict[str, int]:
    if not args.distributed:
        return {"rank": 0, "local_rank": 0, "world_size": 1}
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"--distributed requires torchrun environment variables, missing: {missing}")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return {
        "rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }


def _select_device(args: argparse.Namespace, distributed_state: dict[str, int]) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device in {"auto", "cuda"} and torch.cuda.is_available():
        local_rank = distributed_state["local_rank"]
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if args.device == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def _share_run_dir(run_dir: Path | None, device: torch.device, distributed: bool) -> Path:
    if not distributed:
        if run_dir is None:
            raise RuntimeError("run_dir was not created")
        return run_dir
    payload = [str(run_dir) if run_dir is not None else None]
    dist.broadcast_object_list(payload, src=0)
    return Path(payload[0])


def _parse_int_tuple(value: str, *, expected_len: int | None = None) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {parsed}")
    return parsed


if __name__ == "__main__":
    main()
