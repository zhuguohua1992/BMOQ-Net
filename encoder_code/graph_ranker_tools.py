# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from mesh_boundary_tools import boundary_mask, build_neighbors
from boundary_regressor_tools import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    candidate_features,
    instance_class_map,
    override_boundary,
)


GRAPH_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "ring2_target_fraction",
    "ring2_current_fraction",
    "ring2_boundary_mean",
    "neighbor_target_probability_mean",
    "neighbor_current_probability_mean",
    "neighbor_semantic_margin_mean",
    "neighbor_reliability_mean",
    "ring2_target_probability_mean",
    "ring2_current_probability_mean",
    "target_neighbor_distance_mean",
    "current_neighbor_distance_mean",
    "target_distance_advantage",
    "local_edge_length_mean",
    "laplacian_norm",
]


def parse_obj_geometry(path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "f" and len(fields) >= 4:
                indices = []
                for token in fields[1:]:
                    value = int(token.split("/", 1)[0])
                    value = len(vertices) + value if value < 0 else value - 1
                    indices.append(value)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def second_ring(index, neighbors):
    adjacent = neighbors[index]
    if len(adjacent) == 0:
        return np.empty(0, dtype=np.int64)
    parts = [neighbors[int(neighbor)] for neighbor in adjacent if len(neighbors[int(neighbor)])]
    if not parts:
        return np.empty(0, dtype=np.int64)
    values = np.unique(np.concatenate(parts).astype(np.int64, copy=False))
    return values[values != int(index)]


def graph_candidate_features(index, current, target, state, probability, reliability,
                             boundary_probability, vertices, neighbors, old_boundary,
                             class_map, ring2):
    base = candidate_features(
        index, current, target, state, probability, reliability,
        boundary_probability, neighbors, old_boundary, class_map,
    )
    adjacent = neighbors[index]
    target_class = class_map[target]
    current_class = class_map[current]
    if len(ring2):
        ring2_target_fraction = float(np.mean(state[ring2] == target))
        ring2_current_fraction = float(np.mean(state[ring2] == current))
        ring2_boundary_mean = float(boundary_probability[ring2].mean())
        ring2_target_probability_mean = float(probability[ring2, target_class].mean())
        ring2_current_probability_mean = float(probability[ring2, current_class].mean())
    else:
        ring2_target_fraction = 0.0
        ring2_current_fraction = 0.0
        ring2_boundary_mean = 0.0
        ring2_target_probability_mean = 0.0
        ring2_current_probability_mean = 0.0

    if len(adjacent):
        neighbor_target_probability = probability[adjacent, target_class]
        neighbor_current_probability = probability[adjacent, current_class]
        neighbor_target_probability_mean = float(neighbor_target_probability.mean())
        neighbor_current_probability_mean = float(neighbor_current_probability.mean())
        neighbor_semantic_margin_mean = float(
            (neighbor_target_probability - neighbor_current_probability).mean()
        )
        neighbor_reliability_mean = float(reliability[adjacent].mean())
        distances = np.linalg.norm(vertices[adjacent] - vertices[index], axis=1)
        target_mask = state[adjacent] == target
        current_mask = state[adjacent] == current
        target_distance = float(distances[target_mask].mean()) if np.any(target_mask) else float(distances.mean())
        current_distance = float(distances[current_mask].mean()) if np.any(current_mask) else float(distances.mean())
        local_edge_length_mean = float(distances.mean())
        laplacian_norm = float(np.linalg.norm(vertices[index] - vertices[adjacent].mean(axis=0)))
    else:
        neighbor_target_probability_mean = 0.0
        neighbor_current_probability_mean = 0.0
        neighbor_semantic_margin_mean = 0.0
        neighbor_reliability_mean = 0.0
        target_distance = 0.0
        current_distance = 0.0
        local_edge_length_mean = 0.0
        laplacian_norm = 0.0
    return base + [
        ring2_target_fraction,
        ring2_current_fraction,
        ring2_boundary_mean,
        neighbor_target_probability_mean,
        neighbor_current_probability_mean,
        neighbor_semantic_margin_mean,
        neighbor_reliability_mean,
        ring2_target_probability_mean,
        ring2_current_probability_mean,
        target_distance,
        current_distance,
        current_distance - target_distance,
        local_edge_length_mean,
        laplacian_norm,
    ]


def extract_scan(pred_path, evidence_path, obj_path, gt_path):
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    labels = np.asarray(pred["labels"], dtype=np.int64)
    instances = np.asarray(pred["instances"], dtype=np.int64)
    gt_labels = np.asarray(gt["labels"], dtype=np.int64)
    vertices, triangles = parse_obj_geometry(obj_path)
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
    features = []
    targets = []
    for index in np.flatnonzero(old_boundary):
        index = int(index)
        current = int(instances[index])
        adjacent = neighbors[index]
        ring2 = second_ring(index, neighbors)
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
                new = override_boundary(affected_index, index, target, instances, neighbors)
                gt_value = bool(gt_boundary[affected_index])
                delta_intersection += int(new and gt_value) - int(old and gt_value)
                delta_union += int(new or gt_value) - int(old or gt_value)
            moved = (intersection + delta_intersection) / float(union + delta_union + 1.0e-8)
            features.append(graph_candidate_features(
                index, current, target, instances, probability, reliability,
                boundary_probability, vertices, neighbors, old_boundary,
                class_map, ring2,
            ))
            targets.append((moved - baseline) * 10000.0)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def new_classifier(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.045, max_iter=280, max_leaf_nodes=63,
        min_samples_leaf=60, l2_regularization=1.5, random_state=int(seed),
    )


def new_regressor(seed):
    return HistGradientBoostingRegressor(
        learning_rate=0.045, max_iter=280, max_leaf_nodes=63,
        min_samples_leaf=60, l2_regularization=1.5, random_state=int(seed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    feature_parts = []
    target_parts = []
    groups = []
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
        feature_parts.append(features)
        target_parts.append(targets)
        groups.extend([patient] * len(targets))
        row = {
            "scan_id": scan_id,
            "candidates": int(len(targets)),
            "positive": int(np.sum(targets > 0)),
            "negative": int(np.sum(targets < 0)),
            "tie": int(np.sum(targets == 0)),
        }
        scan_rows.append(row)
        print("TRAIN_EXTRACT " + json.dumps(row, sort_keys=True), flush=True)

    x = np.concatenate(feature_parts, axis=0)
    y_gain = np.concatenate(target_parts, axis=0)
    y_positive = y_gain > 0
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    folds = min(4, len(unique_groups))
    oof_probability = np.zeros(len(y_gain), dtype=np.float64)
    oof_gain = np.zeros(len(y_gain), dtype=np.float64)
    classifier_weight = 1.0 + np.minimum(np.abs(y_gain), 5.0)
    regressor_weight = 1.0 + 2.0 * np.minimum(np.abs(y_gain), 5.0)
    for fold, (train_index, valid_index) in enumerate(
        GroupKFold(n_splits=folds).split(x, y_positive, groups), start=1
    ):
        classifier = new_classifier(args.seed + fold)
        regressor = new_regressor(args.seed + 100 + fold)
        classifier.fit(x[train_index], y_positive[train_index], sample_weight=classifier_weight[train_index])
        regressor.fit(x[train_index], y_gain[train_index], sample_weight=regressor_weight[train_index])
        oof_probability[valid_index] = classifier.predict_proba(x[valid_index])[:, 1]
        oof_gain[valid_index] = regressor.predict(x[valid_index])
        print("TRAIN_FOLD fold={} train={} valid={}".format(
            fold, len(train_index), len(valid_index)
        ), flush=True)

    oof_score = oof_probability * np.maximum(oof_gain, 0.0)
    thresholds = np.unique(np.concatenate((
        np.asarray([0.0], dtype=np.float64),
        np.percentile(oof_score, np.linspace(50.0, 99.95, 700)),
    )))
    choices = []
    for threshold in thresholds:
        selected = oof_score >= threshold
        if int(selected.sum()) < 50:
            continue
        positive = int(np.sum(y_gain[selected] > 0))
        negative = int(np.sum(y_gain[selected] < 0))
        precision = positive / float(max(1, positive + negative))
        choices.append({
            "threshold": float(threshold),
            "selected": int(selected.sum()),
            "positive": positive,
            "negative": negative,
            "precision_nonzero": precision,
            "sum_target_scaled": float(y_gain[selected].sum()),
            "mean_target_scaled": float(y_gain[selected].mean()),
            "mean_probability": float(oof_probability[selected].mean()),
            "mean_predicted_gain": float(oof_gain[selected].mean()),
        })
    eligible = [
        row for row in choices
        if row["precision_nonzero"] >= 0.75 and row["sum_target_scaled"] > 0
    ]
    if not eligible:
        eligible = [row for row in choices if row["sum_target_scaled"] > 0]
    best = max(eligible, key=lambda row: (
        row["sum_target_scaled"], row["precision_nonzero"], -row["selected"]
    ))

    classifier = new_classifier(args.seed)
    regressor = new_regressor(args.seed + 100)
    classifier.fit(x, y_positive, sample_weight=classifier_weight)
    regressor.fit(x, y_gain, sample_weight=regressor_weight)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "classifier": classifier,
        "regressor": regressor,
        "feature_names": GRAPH_FEATURE_NAMES,
        "oof_threshold": best["threshold"],
        "target_scale": 10000.0,
        "seed": args.seed,
        "score_formula": "p_positive * max(predicted_gain, 0)",
    }, str(args.output_model))
    summary = {
        "status": "PASS",
        "train_scans": len(scan_rows),
        "train_patients": len(unique_groups),
        "candidate_count": int(len(y_gain)),
        "positive_count": int(np.sum(y_positive)),
        "negative_count": int(np.sum(y_gain < 0)),
        "tie_count": int(np.sum(y_gain == 0)),
        "feature_names": GRAPH_FEATURE_NAMES,
        "oof_selection": best,
        "top_oof_choices": sorted(
            choices,
            key=lambda row: (row["sum_target_scaled"], row["precision_nonzero"]),
            reverse=True,
        )[:40],
        "scan_rows": scan_rows,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TRAIN_TRAIN_COMPLETE " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit("internal dependency; invoke launch/train_one_epoch.sh")


