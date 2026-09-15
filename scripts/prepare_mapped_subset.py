#!/usr/bin/env python3
# 本文件用于准备完整模型权重可覆盖的扫描及其边界证据。
import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=34)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    prediction_output = args.output_root / "predictions"
    evidence_output = args.output_root / "evidence"
    prediction_output.mkdir(parents=True)
    evidence_output.mkdir(parents=True)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["mapping"]
    selected = []
    for scan_id in sorted(mapping):
        prediction = args.prediction_root / f"{scan_id}.json"
        evidence = args.evidence_root / f"{scan_id}.npz"
        if prediction.is_file() and evidence.is_file():
            link_or_copy(prediction, prediction_output / prediction.name)
            link_or_copy(evidence, evidence_output / evidence.name)
            selected.append(scan_id)
    if len(selected) != args.expected_count:
        raise RuntimeError(
            f"映射扫描数量不符，期望 {args.expected_count}，实际 {len(selected)}。"
        )
    print(json.dumps({"mapped_scan_count": len(selected), "scan_ids": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
