"""从 SEG-Y 数据准备 SCRN 训练 patch 的命令行入口。

本脚本只负责数据准备，不训练模型、不测试模型。默认不会被本轮自动运行。
"""

from __future__ import annotations

import argparse

from SCRN_BRECQ_app.scrn_repro.data import PatchExtractionConfig, collect_patches_from_segy_dir, save_patches_as_npy
from SCRN_BRECQ_app.scrn_repro.utils import require_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SCRN clean patch .npy files from SEG-Y data.")
    parser.add_argument("--input-dir", required=True, help="包含 .segy/.sgy 文件的目录")
    parser.add_argument("--output-dir", required=True, help="保存 patch .npy 文件的目录")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    parser.add_argument("--stride", type=int, nargs=2, default=(48, 48), metavar=("H", "W"))
    parser.add_argument("--augment-times", type=int, default=0, help="每个原始 patch 额外增强次数")
    parser.add_argument("--max-patches", type=int, default=None, help="最多保存多少个 patch")
    parser.add_argument("--min-std", type=float, default=1e-3, help="过滤低方差 patch 的阈值")
    parser.add_argument("--jump", type=int, default=1, help="炮集读取间隔")
    parser.add_argument("--seed", type=int, default=None, help="增强随机种子")
    parser.add_argument("--prefix", default="patch", help="输出 .npy 文件名前缀")
    parser.add_argument("--no-normalize", action="store_true", help="关闭最大绝对值归一化")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = require_directory(args.input_dir, "SEG-Y input directory")
    config = PatchExtractionConfig(
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        augment_times=args.augment_times,
        max_patches=args.max_patches,
        min_std=args.min_std,
        normalize=not args.no_normalize,
        jump=args.jump,
    )
    patches = collect_patches_from_segy_dir(input_dir, config=config, seed=args.seed)
    save_patches_as_npy(patches, args.output_dir, prefix=args.prefix)
    print(f"Saved {len(patches)} patch files to {args.output_dir}")


if __name__ == "__main__":
    main()

