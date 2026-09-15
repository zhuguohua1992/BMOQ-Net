# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from mesh_boundary_tools import (
    boundary_mask,
    build_neighbors,
    fdi_to_internal,
    modal,
    parse_obj,
)


FEATURE_NAMES = [
    "semantic_margin",
    "target_probability",
    "current_probability",
    "reliability",
    "boundary_probability",
    "target_neighbor_fraction",
    "target_neighbor_count",
    "degree",
    "boundary_nll_delta",
    "background_to_tooth",
    "tooth_to_background",
    "tooth_to_tooth",
    "neighbor_boundary_mean",
    "neighbor_boundary_max",
    "current_neighbor_fraction",
    "semantic_entropy",
    "target_is_argmax",
]


def override_boundary(index, moved_index, target, state, neighbors):
    value = target if index == moved_index else state[index]
    for neighbor in neighbors[index]:
        neighbor = int(neighbor)
        other = target if neighbor == moved_index else state[neighbor]
        if value != other:
            return True
    return False


def boundary_nll_delta(index, target, state, neighbors, old_boundary, probability):
    affected = np.concatenate((np.asarray([index], dtype=np.int64), neighbors[index]))
    value = 0.0
    for affected_index in affected:
        affected_index = int(affected_index)
        old = bool(old_boundary[affected_index])
        new = override_boundary(affected_index, index, target, state, neighbors)
        p = float(np.clip(probability[affected_index], 1.0e-4, 1.0 - 1.0e-4))
        value += -math.log(p if new else 1.0 - p) + math.log(p if old else 1.0 - p)
    return value


def instance_class_map(instances, labels, jaw):
    result = {}
    for instance_id in sorted(int(value) for value in np.unique(instances)):
        mask = instances == instance_id
        fdi = 0 if instance_id == 0 else modal(labels[mask])
        result[instance_id] = fdi_to_internal(fdi, jaw)
    return result


def candidate_features(index, current, target, state, probability, reliability,
                       boundary_probability, neighbors, old_boundary, class_map):
    adjacent = neighbors[index]
    current_class = class_map[current]
    target_class = class_map[target]
    p_current = float(probability[index, current_class])
    p_target = float(probability[index, target_class])
    target_count = int(np.count_nonzero(state[adjacent] == target))
    current_count = int(np.count_nonzero(state[adjacent] == current))
    degree = max(1, len(adjacent))
    entropy = float(-(probability[index] * np.log(np.clip(probability[index], 1.0e-8, 1.0))).sum())
    values = [
        p_target - p_current,
        p_target,
        p_current,
        float(reliability[index]),
        float(boundary_probability[index]),
        target_count / float(degree),
        float(target_count),
        float(degree),
        boundary_nll_delta(index, target, state, neighbors, old_boundary, boundary_probability),
        float(current == 0 and target != 0),
        float(current != 0 and target == 0),
        float(current != 0 and target != 0),
        float(boundary_probability[adjacent].mean()) if len(adjacent) else 0.0,
        float(boundary_probability[adjacent].max()) if len(adjacent) else 0.0,
        current_count / float(degree),
        entropy,
        float(int(np.argmax(probability[index])) == target_class),
    ]
    return values


def extract_scan(pred_path, evidence_path, obj_path, gt_path):
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    labels = np.asarray(pred["labels"], dtype=np.int64)
    instances = np.asarray(pred["instances"], dtype=np.int64)
    gt_labels = np.asarray(gt["labels"], dtype=np.int64)
    _, triangles = parse_obj(obj_path)
    neighbors = build_neighbors(len(labels), triangles)
    old_boundary = boundary_mask(instances, neighbors)
    gt_boundary = boundary_mask(gt_labels, neighbors)
    intersection = int(np.logical_and(old_boundary, gt_boundary).sum())
    union = int(np.logical_or(old_boundary, gt_boundary).sum())
    baseline = intersection / float(union + 1.0e-8)
    with np.load(str(evidence_path)) as evidence:
        probability = evidence["semantic_probability"].astype(np.float64)
        reliability = evidence["reliability"].astype(np.float64)
        boundary_probability = evidence["boundary_probability"].astype(np.float64)
    class_map = instance_class_map(instances, labels, pred["jaw"])
    features, targets = [], []
    for index in np.flatnonzero(old_boundary):
        index = int(index)
        current = int(instances[index])
        adjacent = neighbors[index]
        for target in np.unique(instances[adjacent]):
            target = int(target)
            if target == current:
                continue
            affected = np.concatenate((np.asarray([index], dtype=np.int64), adjacent))
            delta_intersection = 0
            delta_union = 0
            for affected_index in affected:
                affected_index = int(affected_index)
                old = bool(old_boundary[affected_index])
                new = override_boundary(
                    affected_index, index, target, instances, neighbors
                )
                gt_value = bool(gt_boundary[affected_index])
                delta_intersection += int(new and gt_value) - int(old and gt_value)
                delta_union += int(new or gt_value) - int(old or gt_value)
            moved = (intersection + delta_intersection) / float(
                union + delta_union + 1.0e-8
            )
            features.append(candidate_features(
                index, current, target, instances, probability, reliability,
                boundary_probability, neighbors, old_boundary, class_map,
            ))
            targets.append((moved - baseline) * 10000.0)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def new_model(seed):
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=int(seed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    all_features, all_targets, all_groups = [], [], []
    scan_rows = []
    for pred_path in sorted(args.pred_root.glob("*.json")):
        scan_id = pred_path.stem
        patient = scan_id.rsplit("_", 1)[0]
        features, targets = extract_scan(
            pred_path,
            args.evidence_root / (scan_id + ".npz"),
            args.obj_root / patient / (scan_id + ".obj"),
            args.json_root / patient / (scan_id + ".json"),
        )
        all_features.append(features)
        all_targets.append(targets)
        all_groups.extend([patient] * len(targets))
        row = {
            "scan_id": scan_id,
            "candidates": int(len(targets)),
            "positive": int((targets > 0).sum()),
            "negative": int((targets < 0).sum()),
        }
        scan_rows.append(row)
        print("TRAIN_EXTRACT " + json.dumps(row, sort_keys=True), flush=True)

    x = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_targets, axis=0)
    groups = np.asarray(all_groups)
    unique_groups = np.unique(groups)
    folds = min(4, len(unique_groups))
    oof = np.zeros(len(y), dtype=np.float64)
    for fold, (train_index, valid_index) in enumerate(
        GroupKFold(n_splits=folds).split(x, y, groups), start=1
    ):
        model = new_model(args.seed + fold)
        model.fit(x[train_index], y[train_index])
        oof[valid_index] = model.predict(x[valid_index])
        print("TRAIN_FOLD fold={} train={} valid={}".format(
            fold, len(train_index), len(valid_index)
        ), flush=True)

    thresholds = np.unique(np.concatenate((
        np.asarray([0.0], dtype=np.float64),
        np.percentile(oof, np.linspace(50.0, 99.9, 500)),
    )))
    choices = []
    for threshold in thresholds:
        selected = oof >= threshold
        count = int(selected.sum())
        if count < 50:
            continue
        positive = int((y[selected] > 0).sum())
        negative = int((y[selected] < 0).sum())
        precision = positive / float(max(1, positive + negative))
        choices.append({
            "threshold": float(threshold),
            "selected": count,
            "positive": positive,
            "negative": negative,
            "precision_nonzero": precision,
            "sum_target_scaled": float(y[selected].sum()),
            "mean_target_scaled": float(y[selected].mean()),
        })
    eligible = [row for row in choices if row["precision_nonzero"] >= 0.60]
    if not eligible:
        eligible = choices
    best = max(
        eligible,
        key=lambda row: (
            row["sum_target_scaled"], row["precision_nonzero"], -row["selected"]
        ),
    )
    final_model = new_model(args.seed)
    final_model.fit(x, y)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "oof_threshold": best["threshold"],
        "target_scale": 10000.0,
        "seed": args.seed,
    }, str(args.output_model))
    summary = {
        "status": "PASS",
        "train_scans": len(scan_rows),
        "train_patients": len(unique_groups),
        "candidate_count": int(len(y)),
        "positive_count": int((y > 0).sum()),
        "negative_count": int((y < 0).sum()),
        "feature_names": FEATURE_NAMES,
        "oof_selection": best,
        "top_oof_choices": sorted(
            choices,
            key=lambda row: (row["sum_target_scaled"], row["precision_nonzero"]),
            reverse=True,
        )[:30],
        "scan_rows": scan_rows,
    }
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TRAIN_TRAIN_COMPLETE " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit("internal dependency; invoke launch/train_one_epoch.sh")


