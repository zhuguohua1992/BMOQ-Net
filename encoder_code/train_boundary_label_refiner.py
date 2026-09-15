# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


SCORE_KEYS = (
    "frontal",
    "frontal_smooth",
    "frontal_r1",
    "frontal_r1_smooth",
    "consensus",
    "max",
    "frontal_sharp",
    "frontal_r1_sharp",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_obj(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                indices = []
                for token in fields[1:]:
                    index = int(token.split("/", 1)[0])
                    index = len(vertices) + index if index < 0 else index - 1
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("OBJ has no vertices/faces: {}".format(path))
    return vertices, faces


def unique_edges(faces):
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def directed_edges(edges):
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    return source, target


def boundary_vertices(labels, edges):
    different = labels[edges[:, 0]] != labels[edges[:, 1]]
    result = np.zeros(len(labels), dtype=bool)
    result[edges[different].reshape(-1)] = True
    return result


def expand_band(seed, edges, hops):
    result = seed.copy()
    for _ in range(hops):
        touches = result[edges[:, 0]] | result[edges[:, 1]]
        result[edges[touches].reshape(-1)] = True
    return result


def neighbor_reduce(values, source, target, vertex_count):
    degree = np.bincount(source, minlength=vertex_count).astype(np.float32)
    total = np.bincount(source, weights=values[target], minlength=vertex_count).astype(np.float32)
    squared = np.bincount(
        source, weights=np.square(values[target]), minlength=vertex_count
    ).astype(np.float32)
    maximum = np.full(vertex_count, -np.inf, dtype=np.float32)
    minimum = np.full(vertex_count, np.inf, dtype=np.float32)
    np.maximum.at(maximum, source, values[target])
    np.minimum.at(minimum, source, values[target])
    empty = degree == 0
    maximum[empty] = values[empty]
    minimum[empty] = values[empty]
    mean = total / np.maximum(degree, 1.0)
    variance = np.maximum(squared / np.maximum(degree, 1.0) - np.square(mean), 0.0)
    return mean, maximum, minimum, np.sqrt(variance), degree


def label_votes(labels, source, target, weights=None):
    label_values = np.unique(labels)
    vertex_count = len(labels)
    if weights is None:
        weights = np.ones(len(source), dtype=np.float32)
    votes = np.empty((vertex_count, len(label_values)), dtype=np.float32)
    for column, label_value in enumerate(label_values):
        votes[:, column] = np.bincount(
            source,
            weights=weights * (labels[target] == label_value),
            minlength=vertex_count,
        )
    return label_values, votes


def feature_names():
    names = ["score_{}".format(key) for key in SCORE_KEYS]
    names += [
        "primary_neighbor_mean",
        "primary_neighbor_max",
        "primary_neighbor_min",
        "primary_neighbor_std",
        "visible_fraction",
        "current_support_1hop",
        "alternative_support_1hop",
        "alternative_minus_current_1hop",
        "disagreement_1hop",
        "current_support_2hop",
        "alternative_support_2hop",
        "alternative_minus_current_2hop",
        "weighted_current_support",
        "weighted_alternative_support",
        "weighted_alternative_minus_current",
        "x_normalized",
        "y_normalized",
        "z_normalized",
        "degree_normalized",
        "mean_edge_length_over_median",
        "current_is_background",
        "alternative_is_background",
        "same_fdi_quadrant",
        "absolute_label_gap",
    ]
    return names


def build_features(vertices, faces, labels, score_path, band_hops):
    edges = unique_edges(faces)
    source, target = directed_edges(edges)
    vertex_count = len(labels)
    boundary = boundary_vertices(labels, edges)
    band = expand_band(boundary, edges, band_hops)
    band_indices = np.flatnonzero(band)

    with np.load(score_path) as data:
        missing = [key for key in SCORE_KEYS if key not in data.files]
        if missing:
            raise KeyError("Missing score keys {} in {}".format(missing, score_path))
        score_arrays = {key: np.asarray(data[key], dtype=np.float32).reshape(-1) for key in SCORE_KEYS}
        visible_count = np.asarray(data["visible_count"], dtype=np.float32).reshape(-1)
    for key, value in score_arrays.items():
        if len(value) != vertex_count:
            raise ValueError("Score length mismatch {} {}".format(key, score_path))

    label_values, votes = label_votes(labels, source, target)
    degree = np.bincount(source, minlength=vertex_count).astype(np.float32)
    vote_fraction = votes / np.maximum(degree[:, None], 1.0)
    lookup = {int(value): index for index, value in enumerate(label_values)}
    current_columns = np.fromiter(
        (lookup[int(value)] for value in labels), dtype=np.int64, count=vertex_count
    )
    rows = np.arange(vertex_count, dtype=np.int64)
    alternative_votes = votes.copy()
    alternative_votes[rows, current_columns] = -1.0
    alternative_columns = np.argmax(alternative_votes, axis=1)
    alternative_labels = label_values[alternative_columns]
    if len(label_values) == 1:
        alternative_labels[:] = labels
        alternative_columns[:] = current_columns

    current_support = vote_fraction[rows, current_columns]
    alternative_support = vote_fraction[rows, alternative_columns]
    two_hop = np.empty_like(vote_fraction)
    for column in range(len(label_values)):
        two_hop[:, column] = np.bincount(
            source, weights=vote_fraction[target, column], minlength=vertex_count
        ) / np.maximum(degree, 1.0)
    current_two_hop = two_hop[rows, current_columns]
    alternative_two_hop = two_hop[rows, alternative_columns]

    primary = score_arrays["frontal_smooth"]
    primary_mean, primary_max, primary_min, primary_std, _ = neighbor_reduce(
        primary, source, target, vertex_count
    )
    edge_image_support = np.maximum(primary[edges[:, 0]], primary[edges[:, 1]])
    edge_weight = np.square(np.clip(1.0 - edge_image_support, 0.02, 1.0)).astype(np.float32)
    directed_weight = np.concatenate((edge_weight, edge_weight))
    _, weighted_votes = label_votes(labels, source, target, directed_weight)
    weighted_degree = np.bincount(
        source, weights=directed_weight, minlength=vertex_count
    ).astype(np.float32)
    weighted_fraction = weighted_votes / np.maximum(weighted_degree[:, None], 1e-8)
    weighted_current = weighted_fraction[rows, current_columns]
    weighted_alternative = weighted_fraction[rows, alternative_columns]

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    coordinate_scale = np.maximum(centered.std(axis=0, keepdims=True), 1e-6)
    normalized_vertices = np.clip(centered / coordinate_scale, -4.0, 4.0)
    undirected_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    median_length = float(np.median(undirected_lengths[undirected_lengths > 0]))
    directed_lengths = np.concatenate((undirected_lengths, undirected_lengths))
    mean_length = np.bincount(
        source, weights=directed_lengths, minlength=vertex_count
    ) / np.maximum(degree, 1.0)

    columns = [score_arrays[key] for key in SCORE_KEYS]
    columns += [
        primary_mean,
        primary_max,
        primary_min,
        primary_std,
        np.clip(visible_count / 8.0, 0.0, 1.0),
        current_support,
        alternative_support,
        alternative_support - current_support,
        1.0 - current_support,
        current_two_hop,
        alternative_two_hop,
        alternative_two_hop - current_two_hop,
        weighted_current,
        weighted_alternative,
        weighted_alternative - weighted_current,
        normalized_vertices[:, 0],
        normalized_vertices[:, 1],
        normalized_vertices[:, 2],
        np.clip(degree / 12.0, 0.0, 2.0),
        np.clip(mean_length / max(median_length, 1e-8), 0.0, 5.0),
        (labels == 0).astype(np.float32),
        (alternative_labels == 0).astype(np.float32),
        ((labels // 10) == (alternative_labels // 10)).astype(np.float32),
        np.clip(np.abs(labels - alternative_labels) / 20.0, 0.0, 3.0).astype(np.float32),
    ]
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(feature_names()):
        raise RuntimeError("Feature count mismatch {} != {}".format(features.shape[1], len(feature_names())))
    return {
        "features": features[band_indices],
        "band_indices": band_indices,
        "alternative_labels": alternative_labels[band_indices],
        "edges": edges,
        "initial_boundary_count": int(boundary.sum()),
        "band_count": int(len(band_indices)),
    }


def scan_paths(stem, obj_root, json_root):
    if not (stem.endswith("_lower") or stem.endswith("_upper")):
        raise ValueError("Bad scan stem {}".format(stem))
    case_id = stem[:-6]
    return obj_root / case_id / (stem + ".obj"), json_root / case_id / (stem + ".json")


def load_cases(pred_roots, score_root, obj_root, json_root, band_hops, read_gt):
    prediction_paths = []
    for root in pred_roots:
        prediction_paths.extend(sorted(root.glob("*.json")))
    names = [path.name for path in prediction_paths]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Prediction roots are empty or contain duplicate names")
    cases = []
    for position, prediction_path in enumerate(sorted(prediction_paths)):
        stem = prediction_path.stem
        obj_path, gt_path = scan_paths(stem, obj_root, json_root)
        score_path = score_root / (stem + ".npz")
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        labels = np.asarray(prediction["labels"], dtype=np.int64).reshape(-1)
        vertices, faces = parse_obj(obj_path)
        if len(vertices) != len(labels):
            raise ValueError("Vertex/prediction length mismatch {}".format(stem))
        built = build_features(vertices, faces, labels, score_path, band_hops)
        ground_truth = None
        if read_gt:
            ground_truth = np.asarray(
                json.loads(gt_path.read_text(encoding="utf-8"))["labels"], dtype=np.int64
            ).reshape(-1)
            if len(ground_truth) != len(labels):
                raise ValueError("GT/prediction length mismatch {}".format(stem))
        cases.append({
            "stem": stem,
            "prediction_path": prediction_path,
            "prediction": prediction,
            "labels": labels,
            "ground_truth": ground_truth,
            "score_path": score_path,
            "obj_path": obj_path,
            **built,
        })
        print(
            "LOAD_CASE {}/{} {} vertices={} band={}".format(
                position + 1, len(prediction_paths), stem, len(labels), built["band_count"]
            ),
            flush=True,
        )
    return cases


def boundary_iou(gt, prediction, edges):
    gt_boundary = boundary_vertices(gt, edges)
    pred_boundary = boundary_vertices(prediction, edges)
    intersection = int(np.logical_and(gt_boundary, pred_boundary).sum())
    union = int(np.logical_or(gt_boundary, pred_boundary).sum())
    return intersection / max(union, 1)


def instance_iou(gt, prediction):
    values = np.unique(prediction)
    values = values[values != 0]
    if not len(values):
        return 0.0
    total = 0.0
    for value in values:
        mask = prediction == value
        gt_values, counts = np.unique(gt[mask], return_counts=True)
        matched = gt_values[np.argmax(counts)]
        gt_mask = gt == matched
        intersection = int(np.logical_and(mask, gt_mask).sum())
        union = int(np.logical_or(mask, gt_mask).sum())
        total += intersection / max(union, 1)
    return total / len(values)


def apply_probability(case, probability, threshold, max_change_fraction):
    labels = case["labels"].copy()
    selected_local = np.flatnonzero(probability >= threshold)
    cap = max(1, int(round(max_change_fraction * len(labels))))
    if len(selected_local) > cap:
        selected_local = selected_local[np.argsort(probability[selected_local])[-cap:]]
    global_indices = case["band_indices"][selected_local]
    labels[global_indices] = case["alternative_labels"][selected_local]
    return labels, int(len(global_indices))


def aggregate_metrics(cases, predictions):
    return {
        "BIoU": float(np.mean([
            boundary_iou(case["ground_truth"], prediction, case["edges"])
            for case, prediction in zip(cases, predictions)
        ])),
        "IoU": float(np.mean([
            instance_iou(case["ground_truth"], prediction)
            for case, prediction in zip(cases, predictions)
        ])),
    }


def write_predictions(cases, probabilities, threshold, cap, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for case, probability in zip(cases, probabilities):
        refined, changed = apply_probability(case, probability, threshold, cap)
        output = dict(case["prediction"])
        output["labels"] = refined.astype(int).tolist()
        if "instances" in output:
            output["instances"] = refined.astype(int).tolist()
        output_path = output_dir / (case["stem"] + ".json")
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        records.append({
            "scan_id": case["stem"],
            "changed_vertices": changed,
            "output": str(output_path),
            "output_sha256": sha256(output_path),
            "prediction_input_sha256": sha256(case["prediction_path"]),
            "score_sha256": sha256(case["score_path"]),
            "obj_sha256": sha256(case["obj_path"]),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--test-pred-root", type=Path, required=True)
    parser.add_argument("--train-score-root", type=Path, required=True)
    parser.add_argument("--val-score-root", type=Path, required=True)
    parser.add_argument("--test-score-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--band-hops", type=int, default=2)
    parser.add_argument("--negative-ratio", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--iou-guard-drop", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError("Run exists: {}".format(args.run_dir))
    args.run_dir.mkdir(parents=True)
    started = time.time()
    rng = np.random.RandomState(args.seed)

    train_cases = load_cases(
        args.train_pred_roots, args.train_score_root, args.obj_root, args.json_root,
        args.band_hops, True,
    )
    train_x = []
    train_y = []
    candidate_stats = []
    for case in train_cases:
        gt_band = case["ground_truth"][case["band_indices"]]
        current_band = case["labels"][case["band_indices"]]
        alternative = case["alternative_labels"]
        valid = (gt_band == current_band) | (gt_band == alternative)
        target = (gt_band == alternative) & (gt_band != current_band)
        train_x.append(case["features"][valid])
        train_y.append(target[valid].astype(np.uint8))
        wrong = gt_band != current_band
        candidate_stats.append({
            "scan_id": case["stem"],
            "band_count": int(len(gt_band)),
            "wrong_in_band": int(wrong.sum()),
            "correctable_by_alternative": int(np.logical_and(wrong, gt_band == alternative).sum()),
            "valid_training_candidates": int(valid.sum()),
        })
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    positive_indices = np.flatnonzero(train_y == 1)
    negative_indices = np.flatnonzero(train_y == 0)
    if not len(positive_indices):
        raise RuntimeError("No positive correction candidates")
    negative_limit = min(len(negative_indices), int(math.ceil(args.negative_ratio * len(positive_indices))))
    sampled_negative = rng.choice(negative_indices, size=negative_limit, replace=False)
    selected = np.concatenate((positive_indices, sampled_negative))
    rng.shuffle(selected)
    fit_x = train_x[selected]
    fit_y = train_y[selected]
    positive_weight = min(max(negative_limit / max(len(positive_indices), 1), 1.0), 20.0)
    sample_weight = np.where(fit_y == 1, positive_weight, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    model.fit(fit_x, fit_y, sample_weight=sample_weight)
    fit_probability = model.predict_proba(fit_x)[:, 1]
    fit_auc = float(roc_auc_score(fit_y, fit_probability))
    fit_ap = float(average_precision_score(fit_y, fit_probability))
    model_path = args.run_dir / "boundary_label_refiner.joblib"
    joblib.dump(model, model_path, compress=3)

    val_cases = load_cases(
        [args.val_pred_root], args.val_score_root, args.obj_root, args.json_root,
        args.band_hops, True,
    )
    val_probabilities = [model.predict_proba(case["features"])[:, 1] for case in val_cases]
    baseline_predictions = [case["labels"] for case in val_cases]
    baseline_metrics = aggregate_metrics(val_cases, baseline_predictions)
    thresholds = [round(value, 2) for value in np.arange(0.10, 0.951, 0.05)]
    caps = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    selection_table = []
    for threshold in thresholds:
        for cap in caps:
            predictions = []
            changes = []
            for case, probability in zip(val_cases, val_probabilities):
                prediction, changed = apply_probability(case, probability, threshold, cap)
                predictions.append(prediction)
                changes.append(changed)
            metrics = aggregate_metrics(val_cases, predictions)
            selection_table.append({
                "threshold": threshold,
                "max_change_fraction": cap,
                "BIoU": metrics["BIoU"],
                "IoU": metrics["IoU"],
                "mean_changed_vertices": float(np.mean(changes)),
                "total_changed_vertices": int(np.sum(changes)),
                "passes_iou_guard": metrics["IoU"] >= baseline_metrics["IoU"] - args.iou_guard_drop,
            })
    eligible = [item for item in selection_table if item["passes_iou_guard"]]
    if not eligible:
        raise RuntimeError("No validation candidate passes IoU guard")
    selected_config = max(
        eligible,
        key=lambda item: (item["BIoU"], item["IoU"], -item["total_changed_vertices"]),
    )

    oracle_predictions = []
    for case in val_cases:
        oracle = case["labels"].copy()
        indices = case["band_indices"]
        gt_band = case["ground_truth"][indices]
        alternative = case["alternative_labels"]
        correctable = (gt_band == alternative) & (gt_band != oracle[indices])
        oracle[indices[correctable]] = alternative[correctable]
        oracle_predictions.append(oracle)
    oracle_metrics = aggregate_metrics(val_cases, oracle_predictions)

    val_records = write_predictions(
        val_cases,
        val_probabilities,
        selected_config["threshold"],
        selected_config["max_change_fraction"],
        args.run_dir / "predictions" / "val",
    )
    test_cases = load_cases(
        [args.test_pred_root], args.test_score_root, args.obj_root, args.json_root,
        args.band_hops, False,
    )
    test_probabilities = [model.predict_proba(case["features"])[:, 1] for case in test_cases]
    test_records = write_predictions(
        test_cases,
        test_probabilities,
        selected_config["threshold"],
        selected_config["max_change_fraction"],
        args.run_dir / "predictions" / "test",
    )

    result = {
        "status": "PASS",
        "schema": "tgn-2d-graph-candidate-label-refiner",
        "scope": "frozen_tgn_plus_frozen_2d_plus_learned_boundary_band_refiner",
        "test_ground_truth_read_by_training_script": False,
        "feature_names": feature_names(),
        "score_keys": list(SCORE_KEYS),
        "arguments": {
            key: ([str(item) for item in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "training": {
            "scan_count": len(train_cases),
            "raw_candidate_count": int(len(train_y)),
            "fit_sample_count": int(len(fit_y)),
            "positive_count": int((fit_y == 1).sum()),
            "negative_count": int((fit_y == 0).sum()),
            "positive_sample_weight": positive_weight,
            "fit_roc_auc": fit_auc,
            "fit_average_precision": fit_ap,
            "actual_iterations": int(model.n_iter_),
            "candidate_stats": candidate_stats,
        },
        "validation": {
            "baseline": baseline_metrics,
            "selected": selected_config,
            "relative_biou_gain": selected_config["BIoU"] / baseline_metrics["BIoU"] - 1.0,
            "oracle_candidate_upper_bound": oracle_metrics,
            "selection_table": selection_table,
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
        },
        "outputs": {
            "validation": val_records,
            "test": test_records,
        },
        "duration_seconds": time.time() - started,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    result_path = args.run_dir / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.run_dir / "COMPLETE").write_text(
        "status=PASS\nresult_sha256={}\n".format(sha256(result_path)), encoding="utf-8"
    )
    print(
        "LABEL_REFINER_STATUS PASS val_biou={:.9f} baseline={:.9f} relative_gain={:.6f} threshold={:.2f} cap={} fit_auc={:.6f} result_sha256={}".format(
            selected_config["BIoU"],
            baseline_metrics["BIoU"],
            selected_config["BIoU"] / baseline_metrics["BIoU"] - 1.0,
            selected_config["threshold"],
            selected_config["max_change_fraction"],
            fit_auc,
            sha256(result_path),
        )
    )


if __name__ == "__main__":
    main()
