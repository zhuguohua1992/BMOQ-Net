#!/usr/bin/env python3
# 本文件用于计算牙齿三维分割的统一评估指标。
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_obj(path):
    vertex_count = 0
    faces = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v":
                vertex_count += 1
            elif fields[0] == "f" and len(fields) >= 4:
                indices = []
                for token in fields[1:]:
                    index = int(token.split("/", 1)[0])
                    if index < 0:
                        index = vertex_count + index
                    else:
                        index -= 1
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertex_count):
        raise ValueError("OBJ face index outside vertex range: {}".format(path))
    return vertex_count, triangles


def build_vertex_neighbors(vertex_count, triangles):
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def get_boundary_labels(labels, neighbors):
    labels = np.asarray(labels).reshape(-1)
    boundary = np.zeros(labels.shape[0], dtype=np.uint8)
    for index, adjacent in enumerate(neighbors):
        for neighbor in adjacent:
            if labels[index] != labels[neighbor]:
                boundary[index] = 1
                break
    return boundary


def calculate_boundary_iou(gt_labels, pred_labels, neighbors):
    gt_boundary = get_boundary_labels(gt_labels, neighbors)
    pred_boundary = get_boundary_labels(pred_labels, neighbors)
    intersection = int(np.logical_and(gt_boundary == 1, pred_boundary == 1).sum())
    union = int(np.logical_or(gt_boundary == 1, pred_boundary == 1).sum())
    return intersection / (union + 1e-8), intersection, union, int(gt_boundary.sum()), int(pred_boundary.sum())


def calculate_local_metrics(gt_labels, pred_sem_labels, pred_ins_labels):
    gt_labels = np.asarray(gt_labels).reshape(-1)
    pred_sem_labels = np.asarray(pred_sem_labels).reshape(-1)
    pred_ins_labels = np.asarray(pred_ins_labels).reshape(-1)
    instance_names = np.unique(pred_ins_labels)
    instance_names = instance_names[instance_names != 0]
    if len(instance_names) == 0:
        return {"IoU": 0.0, "F1": 0.0, "ACC": 0.0, "SEM_ACC": 0.0, "predicted_instance_count": 0}

    totals = {"IoU": 0.0, "F1": 0.0, "ACC": 0.0, "SEM_ACC": 0.0}
    for instance_name in instance_names:
        instance_mask = pred_ins_labels == int(instance_name)
        gt_names, gt_counts = np.unique(gt_labels[instance_mask], return_counts=True)
        gt_name = gt_names[np.argmax(gt_counts)]
        gt_mask = gt_labels == gt_name

        true_positive = int(np.count_nonzero(gt_mask * instance_mask))
        false_negative = int(np.count_nonzero(gt_mask * np.invert(instance_mask)))
        false_positive = int(np.count_nonzero(np.invert(gt_mask) * instance_mask))
        true_negative = int(np.count_nonzero(np.invert(gt_mask) * np.invert(instance_mask)))
        accuracy = (true_positive + true_negative) / (false_positive + true_positive + false_negative + true_negative + 1e-8)
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = true_positive / (false_positive + true_positive + false_negative + 1e-8)

        sem_names, sem_counts = np.unique(pred_sem_labels[instance_mask], return_counts=True)
        sem_name = sem_names[np.argmax(sem_counts)]
        totals["IoU"] += float(iou)
        totals["F1"] += float(f1)
        totals["ACC"] += float(accuracy)
        totals["SEM_ACC"] += float(sem_name == gt_name)

    count = len(instance_names)
    totals = {key: value / count for key, value in totals.items()}
    totals["predicted_instance_count"] = int(count)
    return totals


def scan_paths(prediction_path, obj_root, json_root):
    stem = prediction_path.stem
    if stem.endswith("_lower"):
        case_id = stem[:-6]
    elif stem.endswith("_upper"):
        case_id = stem[:-6]
    else:
        raise ValueError("Prediction name must end in _lower or _upper: {}".format(prediction_path.name))
    return case_id, obj_root / case_id / (stem + ".obj"), json_root / case_id / (stem + ".json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    prediction_paths = sorted(args.pred_root.glob("*.json"))
    if not prediction_paths:
        raise RuntimeError("No prediction JSON files under {}".format(args.pred_root))

    rows = []
    errors = []
    for prediction_path in prediction_paths:
        case_id, obj_path, gt_path = scan_paths(prediction_path, args.obj_root, args.json_root)
        try:
            vertex_count, triangles = parse_obj(obj_path)
            gt_data = load_json(gt_path)
            prediction_data = load_json(prediction_path)
            gt_labels = np.asarray(gt_data["labels"]).reshape(-1)
            pred_labels = np.asarray(prediction_data["labels"]).reshape(-1)
            pred_instances = np.asarray(prediction_data.get("instances", [])).reshape(-1)
            lengths = {
                "obj_vertices": int(vertex_count),
                "gt_labels": int(len(gt_labels)),
                "pred_labels": int(len(pred_labels)),
                "pred_instances": int(len(pred_instances)),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError("length mismatch {}".format(lengths))

            
            
            metrics = calculate_local_metrics(gt_labels, pred_labels, pred_labels)
            neighbors = build_vertex_neighbors(vertex_count, triangles)
            biou, intersection, union, gt_boundary_count, pred_boundary_count = calculate_boundary_iou(
                gt_labels, pred_labels, neighbors
            )
            metrics["BIoU"] = float(biou)
            row = {
                "base_name": prediction_path.stem,
                "case_id": case_id,
                **metrics,
                "vertex_count": int(vertex_count),
                "triangle_count": int(len(triangles)),
                "boundary_intersection": intersection,
                "boundary_union": union,
                "gt_boundary_count": gt_boundary_count,
                "pred_boundary_count": pred_boundary_count,
                "prediction_label_values": [int(value) for value in np.unique(pred_labels)],
                "prediction_instance_values": [int(value) for value in np.unique(pred_instances)],
                "labels_equal_instances": bool(np.array_equal(pred_labels, pred_instances)),
                "lengths": lengths,
                "obj_sha256": sha256(obj_path),
                "gt_sha256": sha256(gt_path),
                "prediction_sha256": sha256(prediction_path),
            }
            rows.append(row)
            print("EVAL_SCAN {} IoU={:.9f} F1={:.9f} ACC={:.9f} SEM_ACC={:.9f} BIoU={:.9f}".format(
                prediction_path.stem, row["IoU"], row["F1"], row["ACC"], row["SEM_ACC"], row["BIoU"]
            ))
        except Exception as exc:
            errors.append({"prediction": str(prediction_path), "error": repr(exc)})

    metric_names = ["IoU", "F1", "ACC", "SEM_ACC", "BIoU"]
    aggregate = {name: float(np.mean([row[name] for row in rows])) for name in metric_names} if rows else {}
    receipt = {
        "status": "PASS" if not errors and len(rows) == len(prediction_paths) else "FAIL",
        "metric_semantics": "uploaded eval_visualize_results_batch.ipynb local five metrics; not official 3DTeethSeg",
        "prediction_count": len(prediction_paths),
        "evaluated_count": len(rows),
        "aggregate_macro_over_scans": aggregate,
        "rows": rows,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_name", "case_id"] + metric_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print("EVAL_AGGREGATE " + json.dumps(aggregate, sort_keys=True))
    print("EVAL_STATUS {} evaluated={} predictions={} errors={}".format(
        receipt["status"], len(rows), len(prediction_paths), len(errors)
    ))
    if receipt["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
