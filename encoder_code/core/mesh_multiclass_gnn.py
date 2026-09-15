# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh_boundary_tools import (
    boundary_mask,
    build_neighbors,
    fdi_to_internal,
)
from boundary_regressor_tools import instance_class_map
from graph_ranker_tools import parse_obj_geometry, second_ring
from boundary_graph_modules import (
    GraphMessageBlock,
    MLP,
    graph_distance,
    soft_iou_loss,
    vertex_normals_and_curvature,
)
from boundary_label_tools import (
    boundary_iou,
    boundary_vertices,
    expand_band,
    instance_iou,
    sha256,
    unique_edges,
)


SCHEMA = "boundary-multiclass-mesh-gnn"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def directed_edges(edges):
    return (
        np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64),
        np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64),
    )


def class_histogram(classes, source, target, vertex_count):
    result = np.zeros((vertex_count, 17), dtype=np.float32)
    np.add.at(result, (source, classes[target]), 1.0)
    degree = np.bincount(source, minlength=vertex_count).astype(np.float32)
    result /= np.maximum(degree[:, None], 1.0)
    return result, degree


def build_target_instances(band_indices, instances, classes, neighbors, class_map):
    result = np.full((len(band_indices), 17), -1, dtype=np.int32)
    for local_index, global_value in enumerate(band_indices):
        index = int(global_value)
        current = int(instances[index])
        result[local_index, int(classes[index])] = current
        adjacent = neighbors[index]
        ring2 = second_ring(index, neighbors)
        pool = np.concatenate((adjacent, adjacent, ring2)) if len(ring2) else np.concatenate((adjacent, adjacent))
        if not len(pool):
            continue
        values, counts = np.unique(instances[pool], return_counts=True)
        best = {}
        for value, count in zip(values, counts):
            instance_id = int(value)
            class_id = int(class_map[instance_id])
            previous = best.get(class_id)
            candidate = (int(count), -instance_id, instance_id)
            if previous is None or candidate > previous:
                best[class_id] = candidate
        for class_id, candidate in best.items():
            result[local_index, class_id] = int(candidate[-1])
    return result


def build_case(prediction_path, evidence_root, obj_root, json_root, band_hops,
               boundary_distance_hops, protect_reference_root=None):
    stem = prediction_path.stem
    patient = stem.rsplit("_", 1)[0]
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    labels = np.asarray(payload["labels"], dtype=np.int64)
    instances = np.asarray(payload["instances"], dtype=np.int64)
    vertices, faces = parse_obj_geometry(obj_root / patient / (stem + ".obj"))
    edges = unique_edges(faces)
    source, target = directed_edges(edges)
    neighbors = build_neighbors(len(instances), faces)
    class_map = instance_class_map(instances, labels, payload["jaw"])
    classes = np.fromiter(
        (class_map[int(value)] for value in instances),
        dtype=np.int64,
        count=len(instances),
    )
    with np.load(str(evidence_root / (stem + ".npz"))) as handle:
        probability = handle["semantic_probability"].astype(np.float32)
        reliability = handle["reliability"].astype(np.float32)
        boundary_probability = handle["boundary_probability"].astype(np.float32)
    coarse_boundary = boundary_mask(instances, neighbors)
    band = expand_band(coarse_boundary, edges, int(band_hops))
    band_indices = np.flatnonzero(band)
    neighbor_histogram, degree = class_histogram(classes, source, target, len(instances))
    two_hop = np.empty_like(neighbor_histogram)
    for class_id in range(17):
        two_hop[:, class_id] = np.bincount(
            source,
            weights=neighbor_histogram[target, class_id],
            minlength=len(instances),
        ) / np.maximum(degree, 1.0)
    current_onehot = np.eye(17, dtype=np.float32)[classes]
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    normalized = (centered / np.maximum(centered.std(axis=0, keepdims=True), 1.0e-6)).astype(np.float32)
    normals, curvature_mean, curvature_max = vertex_normals_and_curvature(
        vertices.astype(np.float32), faces, edges
    )
    coarse_distance = graph_distance(coarse_boundary, edges, max(int(boundary_distance_hops), 1))
    entropy = -np.sum(probability * np.log(np.clip(probability, 1.0e-8, 1.0)), axis=1)
    columns = [
        probability,
        current_onehot,
        neighbor_histogram,
        two_hop,
        reliability[:, None],
        boundary_probability[:, None],
        normalized,
        normals,
        curvature_mean[:, None],
        curvature_max[:, None],
        np.clip(coarse_distance / max(int(boundary_distance_hops), 1), 0.0, 1.0)[:, None].astype(np.float32),
        entropy[:, None].astype(np.float32),
        probability.max(axis=1, keepdims=True),
    ]
    features = np.concatenate(columns, axis=1).astype(np.float32)[band_indices]
    global_to_local = np.full(len(instances), -1, dtype=np.int64)
    global_to_local[band_indices] = np.arange(len(band_indices), dtype=np.int64)
    inside = (global_to_local[edges[:, 0]] >= 0) & (global_to_local[edges[:, 1]] >= 0)
    local_edges = np.column_stack(
        (global_to_local[edges[inside, 0]], global_to_local[edges[inside, 1]])
    ).astype(np.int64)
    ground_truth = np.asarray(
        json.loads((json_root / patient / (stem + ".json")).read_text(encoding="utf-8"))["labels"],
        dtype=np.int64,
    )
    gt_classes = np.fromiter(
        (fdi_to_internal(int(value), payload["jaw"]) for value in ground_truth),
        dtype=np.int64,
        count=len(ground_truth),
    )
    gt_boundary = boundary_vertices(ground_truth, edges)
    protected_local = np.zeros(len(band_indices), dtype=np.bool_)
    if protect_reference_root is not None:
        reference = json.loads(
            (protect_reference_root / prediction_path.name).read_text(encoding="utf-8")
        )
        reference_instances = np.asarray(reference["instances"], dtype=np.int64)
        protected_local = (instances != reference_instances)[band_indices]
    return {
        "stem": stem,
        "jaw": payload["jaw"],
        "payload": payload,
        "features": features,
        "band_indices": band_indices,
        "local_edges": local_edges,
        "edges": edges,
        "instances": instances,
        "labels": labels,
        "current_classes": classes[band_indices],
        "target_classes": gt_classes[band_indices],
        "gt_labels": ground_truth,
        "gt_boundary_local": gt_boundary[band_indices].astype(np.float32),
        "target_instances": build_target_instances(
            band_indices, instances, classes, neighbors, class_map
        ),
        "neighbor_support": neighbor_histogram[band_indices],
        "protected_local": protected_local,
    }


def _worker(task):
    return build_case(*task)


def load_cases(pred_root, evidence_root, obj_root, json_root, band_hops,
               boundary_distance_hops, protect_reference_root, workers):
    paths = sorted(pred_root.glob("*.json"))
    tasks = [
        (
            path, evidence_root, obj_root, json_root, band_hops,
            boundary_distance_hops, protect_reference_root,
        )
        for path in paths
    ]
    cases = []
    context = mp.get_context("fork")
    with context.Pool(processes=int(workers)) as pool:
        for position, case in enumerate(pool.imap(_worker, tasks, chunksize=1), 1):
            cases.append(case)
            print(
                "TRAIN_LOAD {}/{} {} band={} edges={}".format(
                    position, len(paths), case["stem"], len(case["band_indices"]),
                    len(case["local_edges"])
                ),
                flush=True,
            )
    return cases


def fit_standardizer(cases):
    count = 0
    total = None
    squared = None
    for case in cases:
        values = case["features"].astype(np.float64)
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        squared = np.square(values).sum(axis=0) if squared is None else squared + np.square(values).sum(axis=0)
        count += len(values)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(squared / max(count, 1) - np.square(mean), 1.0e-8))
    std[std < 1.0e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(cases, mean, std):
    for case in cases:
        case["features"] = np.clip((case["features"] - mean) / std, -8.0, 8.0).astype(np.float32)


class MultiClassMeshGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            [GraphMessageBlock(hidden_dim, dropout) for _ in range(int(layers))]
        )
        self.context = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 17)
        self.boundary_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, source, target):
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden, source, target)
        context = self.context(
            torch.cat((hidden.mean(dim=0, keepdim=True), hidden.max(dim=0, keepdim=True)[0]), dim=1)
        )
        hidden = self.norm(hidden + context)
        return self.class_head(hidden), self.boundary_head(hidden).squeeze(1)


def tensors(case, device):
    edge = case["local_edges"]
    source = np.concatenate((edge[:, 0], edge[:, 1])).astype(np.int64)
    target = np.concatenate((edge[:, 1], edge[:, 0])).astype(np.int64)
    return {
        "x": torch.from_numpy(case["features"]).to(device),
        "source": torch.from_numpy(source).to(device),
        "target": torch.from_numpy(target).to(device),
        "edge_u": torch.from_numpy(edge[:, 0]).to(device),
        "edge_v": torch.from_numpy(edge[:, 1]).to(device),
        "current": torch.from_numpy(case["current_classes"]).to(device),
        "target_class": torch.from_numpy(case["target_classes"]).to(device),
        "gt_boundary": torch.from_numpy(case["gt_boundary_local"]).to(device),
    }


def calculate_loss(logits, boundary_logit, values):
    target = values["target_class"]
    current = values["current"]
    log_probability = F.log_softmax(logits, dim=1)
    negative_log_likelihood = -log_probability.gather(1, target[:, None]).squeeze(1)
    uniform_smoothing = -log_probability.mean(dim=1)
    raw = 0.98 * negative_log_likelihood + 0.02 * uniform_smoothing
    probability = torch.softmax(logits, dim=1)
    target_probability = probability.gather(1, target[:, None]).squeeze(1)
    focal = torch.pow(1.0 - target_probability, 1.5)
    changed = target != current
    weight = torch.where(changed, torch.full_like(raw, 6.0), torch.ones_like(raw))
    semantic = (raw * focal * weight).mean()
    gt_boundary = values["gt_boundary"]
    boundary_weight = torch.where(gt_boundary > 0.5, torch.full_like(gt_boundary, 2.0), torch.ones_like(gt_boundary))
    auxiliary = F.binary_cross_entropy_with_logits(boundary_logit, gt_boundary, weight=boundary_weight)
    u, v = values["edge_u"], values["edge_v"]
    edge_probability = (1.0 - (probability[u] * probability[v]).sum(dim=1)).clamp(1.0e-5, 1.0 - 1.0e-5)
    edge_target = (target[u] != target[v]).float()
    edge_loss = F.binary_cross_entropy(edge_probability, edge_target) + soft_iou_loss(edge_probability, edge_target)
    total = semantic + 0.20 * auxiliary + 1.0 * edge_loss
    return total, {
        "total": float(total.detach().cpu()),
        "semantic": float(semantic.detach().cpu()),
        "auxiliary": float(auxiliary.detach().cpu()),
        "edge": float(edge_loss.detach().cpu()),
        "change_fraction": float(changed.float().mean().detach().cpu()),
    }


def predict(model, cases, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for case in cases:
            value = tensors(case, device)
            logits, boundary = model(value["x"], value["source"], value["target"])
            outputs.append({
                "probability": torch.softmax(logits, dim=1).cpu().numpy(),
                "boundary": torch.sigmoid(boundary).cpu().numpy(),
            })
    return outputs


def components(selected, target_instances, current_instances, local_edges, minimum_size):
    proposals = {int(index): (int(current_instances[index]), int(target_instances[index])) for index in np.flatnonzero(selected)}
    adjacency = {index: [] for index in proposals}
    for a, b in local_edges:
        a, b = int(a), int(b)
        if a in proposals and b in proposals and proposals[a] == proposals[b]:
            adjacency[a].append(b)
            adjacency[b].append(a)
    remaining = set(proposals)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        group = []
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in adjacency[index]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        if len(group) >= int(minimum_size):
            result.append(group)
    return result


def apply_spec(case, prediction, spec):
    probability = prediction["probability"]
    current_class = case["current_classes"]
    target_instances = case["target_instances"]
    masked = probability.copy()
    masked[target_instances < 0] = -1.0
    masked[np.arange(len(masked)), current_class] = -1.0
    target_class = np.argmax(masked, axis=1)
    target_probability = masked[np.arange(len(masked)), target_class]
    current_probability = probability[np.arange(len(probability)), current_class]
    margin = target_probability - current_probability
    support = case["neighbor_support"][np.arange(len(masked)), target_class]
    target_instance = target_instances[np.arange(len(masked)), target_class]
    eligible = (
        (target_instance >= 0)
        & (target_probability >= float(spec["probability_minimum"]))
        & (margin >= float(spec["margin_minimum"]))
        & (support >= float(spec["support_minimum"]))
        & ~case["protected_local"]
    )
    score = target_probability * np.maximum(margin, 0.0) * (0.10 + support)
    current_instances = case["instances"][case["band_indices"]]
    groups = components(
        eligible,
        target_instance,
        current_instances,
        case["local_edges"],
        int(spec["minimum_component_size"]),
    )
    groups.sort(key=lambda group: (float(np.mean(score[group])), -len(group)), reverse=True)
    cap = max(1, int(round(float(spec["cap_fraction"]) * len(case["instances"]))))
    chosen = []
    for group in groups:
        if len(chosen) + len(group) > cap:
            continue
        chosen.extend(group)
    output = case["instances"].copy()
    for local_index in chosen:
        output[int(case["band_indices"][local_index])] = int(target_instance[local_index])
    mapping = {}
    for instance_id in np.unique(case["instances"]):
        mask = case["instances"] == int(instance_id)
        values, counts = np.unique(case["labels"][mask], return_counts=True)
        mapping[int(instance_id)] = int(values[int(np.argmax(counts))])
    lookup = np.zeros(int(output.max()) + 1, dtype=np.int64)
    for instance_id, label in mapping.items():
        lookup[instance_id] = label
    output_labels = lookup[output]
    return output_labels, output, len(chosen)


def evaluate(cases, predictions, spec):
    outputs = []
    rows = []
    for case, prediction in zip(cases, predictions):
        labels, instances, changed = apply_spec(case, prediction, spec)
        outputs.append((labels, instances))
        rows.append(changed)
    metrics = {
        "BIoU": float(np.mean([boundary_iou(case["gt_labels"], value[0], case["edges"]) for case, value in zip(cases, outputs)])),
        "IoU": float(np.mean([instance_iou(case["gt_labels"], value[0]) for case, value in zip(cases, outputs)])),
    }
    return metrics, outputs, rows


def select_validation(cases, predictions, selection_mode="full"):
    specs = []
    if selection_mode == "fast":
        probabilities = (0.50, 0.70, 0.90)
        margins = (0.0, 0.10, 0.20)
        supports = (0.10, 0.30)
        caps = (0.001, 0.002, 0.005)
        sizes = (1,)
    else:
        probabilities = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
        margins = (0.0, 0.10, 0.20, 0.30)
        supports = (0.10, 0.25, 0.40)
        caps = (0.0005, 0.001, 0.002, 0.005)
        sizes = (1, 2)
    for probability in probabilities:
        for margin in margins:
            for support in supports:
                for cap in caps:
                    for size in sizes:
                        specs.append({
                            "probability_minimum": probability,
                            "margin_minimum": margin,
                            "support_minimum": support,
                            "cap_fraction": cap,
                            "minimum_component_size": size,
                        })
    baseline = {
        "BIoU": float(np.mean([boundary_iou(case["gt_labels"], case["labels"], case["edges"]) for case in cases])),
        "IoU": float(np.mean([instance_iou(case["gt_labels"], case["labels"]) for case in cases])),
    }
    table = []
    for spec in specs:
        metrics, _, changed = evaluate(cases, predictions, spec)
        table.append({**spec, **metrics, "total_changed": int(sum(changed)), "passes_iou_guard": metrics["IoU"] >= baseline["IoU"] - 0.001})
    eligible = [row for row in table if row["passes_iou_guard"]]
    selected = max(eligible, key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed"]))
    return baseline, selected, table


def write_predictions(cases, predictions, spec, output_root):
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, prediction in zip(cases, predictions):
        labels, instances, changed = apply_spec(case, prediction, spec)
        payload = dict(case["payload"])
        payload["labels"] = labels.astype(int).tolist()
        payload["instances"] = instances.astype(int).tolist()
        path = output_root / (case["stem"] + ".json")
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        rows.append({"scan_id": case["stem"], "changed": int(changed), "sha256": sha256(path)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-root", type=Path, required=True)
    parser.add_argument("--train-evidence-root", type=Path, required=True)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--val-evidence-root", type=Path, required=True)
    parser.add_argument("--val-protect-reference-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--band-hops", type=int, default=1)
    parser.add_argument("--boundary-distance-hops", type=int, default=8)
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260865)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-mode", choices=("full", "fast"), default="full")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--start-epoch", type=int, default=1)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    started = time.time()
    train_cases = load_cases(args.train_pred_root, args.train_evidence_root, args.obj_root, args.json_root, args.band_hops, args.boundary_distance_hops, None, args.load_workers)
    val_cases = load_cases(args.val_pred_root, args.val_evidence_root, args.obj_root, args.json_root, args.band_hops, args.boundary_distance_hops, args.val_protect_reference_root, min(args.load_workers, 4))
    mean, std = fit_standardizer(train_cases)
    standardize(train_cases, mean, std)
    standardize(val_cases, mean, std)
    set_seed(args.seed)
    device = torch.device(args.device)
    model = MultiClassMeshGNN(train_cases[0]["features"].shape[1], args.hidden_dim, args.graph_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.start_epoch + 1, 1), eta_min=args.learning_rate * 0.05)
    order_rng = np.random.RandomState(args.seed)
    for _ in range(max(int(args.start_epoch) - 1, 0)):
        order_rng.permutation(len(train_cases))
    history = []
    best = None
    best_path = args.run_dir / "best_model.pt"
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        best = {
            "epoch": int(checkpoint.get("epoch", args.start_epoch - 1)),
            "selected": checkpoint["selected"],
            "top": [],
        }
        torch.save(checkpoint, best_path)
        print("TRAIN_RESUME epoch={} checkpoint={}".format(best["epoch"], args.resume_checkpoint), flush=True)
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for position, case_index in enumerate(order_rng.permutation(len(train_cases)), 1):
            case = train_cases[int(case_index)]
            value = tensors(case, device)
            optimizer.zero_grad()
            logits, boundary = model(value["x"], value["source"], value["target"])
            loss, row = calculate_loss(logits, boundary, value)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(row)
            if position % 50 == 0 or position == len(train_cases):
                print("TRAIN_TRAIN epoch={} {}/{} loss={:.6f}".format(epoch, position, len(train_cases), row["total"]), flush=True)
        scheduler.step()
        predictions = predict(model, val_cases, device)
        baseline, selected, table = select_validation(val_cases, predictions, args.selection_mode)
        row = {"epoch": epoch, "loss": float(np.mean([item["total"] for item in losses])), "baseline": baseline, "selected": selected}
        history.append(row)
        print("TRAIN_EPOCH {} loss={:.6f} val_biou={:.9f} baseline={:.9f} val_iou={:.9f} spec={}".format(epoch, row["loss"], selected["BIoU"], baseline["BIoU"], selected["IoU"], json.dumps({key: selected[key] for key in ("probability_minimum", "margin_minimum", "support_minimum", "cap_fraction", "minimum_component_size")}, sort_keys=True)), flush=True)
        if best is None or (selected["BIoU"], selected["IoU"], -selected["total_changed"]) > (best["selected"]["BIoU"], best["selected"]["IoU"], -best["selected"]["total_changed"]):
            best = {"epoch": epoch, "selected": selected, "top": sorted(table, key=lambda item: (item["BIoU"], item["IoU"]), reverse=True)[:30]}
            torch.save({"schema": SCHEMA, "epoch": epoch, "state_dict": model.state_dict(), "input_dim": train_cases[0]["features"].shape[1], "hidden_dim": args.hidden_dim, "graph_layers": args.graph_layers, "dropout": args.dropout, "mean": mean, "std": std, "selected": selected}, best_path)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict(model, val_cases, device)
    outputs = write_predictions(val_cases, predictions, best["selected"], args.run_dir / "predictions" / "val")
    result = {"status": "PASS", "schema": SCHEMA, "best": best, "history": history, "parameter_count": int(sum(p.numel() for p in model.parameters())), "model_sha256": sha256(best_path), "outputs": outputs, "duration_seconds": time.time() - started}
    (args.run_dir / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRAIN_COMPLETE best_epoch={} val_biou={:.9f}".format(best["epoch"], best["selected"]["BIoU"]), flush=True)


if __name__ == "__main__":
    main()


