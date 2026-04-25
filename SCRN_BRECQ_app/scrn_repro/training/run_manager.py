"""实验 run 目录和记录文件管理。

训练和测试会产生较多 checkpoint、预测和图片文件，因此所有产物都放入独立 run 目录，
并用 JSON/CSV/Markdown 记录可复现实验信息。
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import torch


def create_run_dir(root: str | Path, *, run_name: str | None = None, timestamp: str | None = None) -> Path:
    """创建形如 `<root>/<timestamp>_<run_name>` 的 run 目录。"""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_name(run_name) if run_name else "run"
    run_dir = Path(root) / f"{stamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_environment() -> dict[str, Any]:
    """收集运行环境信息，写入 config 便于追踪复现实验。"""
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_output(["git", "branch", "--show-current"]),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写入带缩进的 JSON 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """追加一行 JSONL 指标记录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_csv_row(path: str | Path, row: Mapping[str, Any], *, fieldnames: Iterable[str]) -> None:
    """追加 CSV 行；文件不存在时自动写表头。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: str | Path, *, title: str, sections: Mapping[str, Any]) -> None:
    """写一个简洁 Markdown summary。"""
    lines = [f"# {title}", ""]
    for section, content in sections.items():
        lines.extend([f"## {section}", ""])
        if isinstance(content, Mapping):
            for key, value in content.items():
                lines.append(f"- {key}: `{value}`")
        else:
            lines.append(str(content))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _safe_name(name: str | None) -> str:
    if not name:
        return "run"
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name.strip())
    return safe or "run"


def _git_output(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()
