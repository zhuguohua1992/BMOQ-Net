#!/usr/bin/env python3
# 本文件用于汇总所有方法在完整测试集上的五项指标。
import argparse
import csv
import json
from pathlib import Path


METRICS = ("IoU", "F1", "SEM_ACC", "ACC", "BIoU")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=180)
    args = parser.parse_args()
    rows = []
    for item in args.method:
        display_name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"评估未通过: {path}")
        if payload.get("prediction_count") != args.expected_count:
            raise RuntimeError(f"预测数量不是 {args.expected_count}: {path}")
        if payload.get("evaluated_count") != args.expected_count or payload.get("errors"):
            raise RuntimeError(f"评估覆盖不完整: {path}")
        values = payload["aggregate_macro_over_scans"]
        rows.append({"Method": display_name, **{name: float(values[name]) for name in METRICS}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Method", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
