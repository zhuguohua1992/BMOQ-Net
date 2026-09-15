# 本文件用于实现完整模型的边界编码、训练或推理。
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.clean_checkpoint)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("Expected a checkpoint dictionary with state_dict")
    if "selected" in checkpoint:
        raise RuntimeError("Input is not the expected metadata-clean export")

    with args.runtime_selection.open("r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    selected = sidecar.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("runtime-selection sidecar has no selected dictionary")

    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["selected"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(runtime_checkpoint, args.output)
    print(f"WROTE_TEMP_RUNTIME_CHECKPOINT={args.output}")


if __name__ == "__main__":
    main()
