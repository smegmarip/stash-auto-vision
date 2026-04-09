"""
Export a trained occlusion classifier checkpoint to ONNX.

Used to convert a PyTorch ``.pth`` checkpoint produced by the occlusion
detection training pipeline into the ``.onnx`` file that the faces-service
runtime (``occlusion_detector.py``) loads via onnxruntime.

This is a reference / distribution copy — the full training pipeline lives
in a separate experiment repository. It's included here so anyone who
retrains the classifier on their own data can produce a drop-in replacement
for ``occlusion_classifier.onnx`` without pulling in the rest of the
training rig.

Usage:

    cd faces-service/training
    pip install -r requirements.txt
    python export_onnx.py \\
        --checkpoint /path/to/best_model.pth \\
        --output     /path/to/occlusion_classifier.onnx \\
        --config     export_config.yaml

The resulting ``.onnx`` can be consumed by the faces-service in two ways:

1. **Via HuggingFace** — upload it to a fork of ``smegmarip/face-recognition``
   and point ``FACES_HF_REPO`` at the fork.
2. **Via local-path override** — mount the file into the container and set
   ``FACES_LOCAL_OCCLUSION_PATH`` to its absolute path. bootstrap.py will
   detect it and skip the HF download.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from models import get_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained occlusion classifier .pth checkpoint to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the exported .onnx file.",
    )
    parser.add_argument(
        "--config",
        default="export_config.yaml",
        help="YAML config file with training.model architecture (default: export_config.yaml).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11, matches what the runtime expects).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = config.get("training", {}).get("model")
    if not model_name:
        raise SystemExit(
            f"Config {config_path} is missing 'training.model' "
            "(must be 'resnet18' or 'convnext_small')."
        )

    logging.info("Loading %s checkpoint from %s", model_name, checkpoint_path)
    model = get_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Accept either a full training checkpoint dict or a raw state_dict.
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)

    logging.info("Exporting to %s (opset=%d)", output_path, args.opset)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    size_mb = output_path.stat().st_size / (1024 ** 2)
    logging.info("Exported successfully: %s (%.1f MB)", output_path, size_mb)


if __name__ == "__main__":
    main()
