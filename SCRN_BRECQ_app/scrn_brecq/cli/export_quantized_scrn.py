"""Export a SCRN-BRECQ checkpoint into a compact packed deployment artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    require_file,
    restore_quantizer_state_shapes,
)
from SCRN_BRECQ_app.scrn_brecq.utils import export_packed_deployment


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Export a SCRN-BRECQ checkpoint as packed deployment files.")
    parser.add_argument("--checkpoint", required=True, help="Path to quantized_scrn_brecq.pth")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to <run_dir>/packed_deployment",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    checkpoint_path = require_file(args.checkpoint, "quantized checkpoint")
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir(checkpoint_path)

    checkpoint = load_quant_checkpoint(checkpoint_path)
    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.cpu()
    quant_model.eval()

    summary = export_packed_deployment(
        quant_model,
        output_dir,
        source_checkpoint_path=checkpoint.get("source_checkpoint"),
        quant_checkpoint_path=checkpoint_path,
        final_quant_state=checkpoint.get("final_quant_state"),
        checkpoint_metadata={
            "checkpoint_stage": checkpoint.get("checkpoint_stage"),
            "quant_config": checkpoint.get("quant_config", {}),
            "model_config": checkpoint.get("model_config", {}),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "raw_deployment_payload_mib": summary["payload"]["raw_deployment_payload_mib"],
                "total_export_file_size_mib": summary["files"]["total_export_file_size_mib"],
                "estimated_packed_model_size_mib": summary["comparison"]["estimated_packed_model_size_mib"],
                "raw_payload_to_estimated_packed_ratio": summary["comparison"]["raw_payload_to_estimated_packed_ratio"],
                "estimated_model_compression_ratio": summary["comparison"]["estimated_model_compression_ratio"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def default_output_dir(checkpoint_path: Path) -> Path:
    """Return <run_dir>/packed_deployment for a checkpoint under <run_dir>/checkpoints."""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "packed_deployment"
    return checkpoint_path.parent / "packed_deployment"


if __name__ == "__main__":
    main()
