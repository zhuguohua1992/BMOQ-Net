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

import mesh_multiclass_gnn as mesh_multiclass


SCHEMA = "boundary-firststage-edge-gated-mesh-gnn"


def add_gain_targets(case):
    labels = case["labels"]
    instances = case["instances"]
    ground_truth = case["gt_labels"]
    target_instances = case["target_instances"]
    band_indices = case["band_indices"]
    edges = case["edges"]

    instance_label = {}
    for instance_id in np.unique(instances):
        values, counts = np.unique(labels[instances == int(instance_id)], return_counts=True)
        instance_label[int(instance_id)] = int(values[int(np.argmax(counts))])
    candidate_label = np.full(target_instances.shape, -1, dtype=np.int64)
    for instance_id in np.unique(target_instances[target_instances >= 0]):
        candidate_label[target_instances == int(instance_id)] = instance_label[int(instance_id)]

    global_to_local = np.full(len(labels), -1, dtype=np.int64)
    global_to_local[band_indices] = np.arange(len(band_indices), dtype=np.int64)
    directed_u = np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64)
    directed_v = np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64)
    local_u = global_to_local[directed_u]
    keep = local_u >= 0
    local_u = local_u[keep]
    global_u = directed_u[keep]
    global_v = directed_v[keep]
    old_boundary = labels[global_u] != labels[global_v]
    true_boundary = ground_truth[global_u] != ground_truth[global_v]
    old_correct = old_boundary == true_boundary
    degree = np.bincount(local_u, minlength=len(band_indices)).astype(np.float32)
    gain = np.zeros(target_instances.shape, dtype=np.float32)
    for class_id in range(17):
        proposed = candidate_label[local_u, class_id]
        valid = proposed >= 0
        if not np.any(valid):
            continue
        new_boundary = proposed[valid] != labels[global_v[valid]]
        delta = (
            (new_boundary == true_boundary[valid]).astype(np.float32)
            - old_correct[valid].astype(np.float32)
        )
        np.add.at(gain[:, class_id], local_u[valid], delta)
    gain /= np.maximum(degree[:, None], 1.0)
    valid = target_instances >= 0
    valid[np.arange(len(valid)), case["current_classes"]] = False
    case["gain_target"] = gain.astype(np.float32)
    case["gain_valid"] = valid
    return case


def build_case(*task):
    return add_gain_targets(mesh_multiclass.build_case(*task))


def _worker(task):
    return build_case(*task)


def load_cases(pred_root, evidence_root, obj_root, json_root, band_hops,
               boundary_distance_hops, protect_reference_root, workers):
    paths = sorted(pred_root.glob("*.json"))
    tasks = [
        (
            path,
            evidence_root,
            obj_root,
            json_root,
            band_hops,
            boundary_distance_hops,
            protect_reference_root,
        )
        for path in paths
    ]
    result = []
    if int(workers) <= 1:
        iterator = map(_worker, tasks)
        for position, value in enumerate(iterator, 1):
            result.append(value)
            print("TRAIN_LOAD {}/{} {} band={} edges={}".format(
                position, len(tasks), value["stem"], len(value["band_indices"]), len(value["local_edges"])
            ), flush=True)
        return result
    with mp.Pool(processes=int(workers)) as pool:
        for position, value in enumerate(pool.imap(_worker, tasks, chunksize=1), 1):
            result.append(value)
            print("TRAIN_LOAD {}/{} {} band={} edges={}".format(
                position, len(tasks), value["stem"], len(value["band_indices"]), len(value["local_edges"])
            ), flush=True)
    return result


class EdgeGatedMessageBlock(nn.Module):
                                                                                       

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.message = mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, source, target):
        neighbor = hidden[target]
        pair = torch.cat((neighbor, neighbor - hidden[source]), dim=1)
        gate = torch.sigmoid(self.gate(pair))
        message = gate * self.message(pair)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, source, message)
        normalizer = torch.zeros(len(hidden), dtype=hidden.dtype, device=hidden.device)
        normalizer.index_add_(0, source, gate.squeeze(1))
        aggregate = aggregate / normalizer.clamp(min=1.0e-3).unsqueeze(1)
        update = self.update(torch.cat((hidden, aggregate), dim=1))
        return self.norm(hidden + self.dropout(update))


class EdgeGatedGainAwareMeshGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.encoder = mesh_multiclass.MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            [EdgeGatedMessageBlock(hidden_dim, dropout) for _ in range(int(layers))]
        )
        self.context = mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 17)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 17),
        )
        self.edge_head = nn.Sequential(
            mesh_multiclass.MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, source, target, edge_u, edge_v):
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden, source, target)
        context = self.context(
            torch.cat((hidden.mean(dim=0, keepdim=True), hidden.max(dim=0, keepdim=True)[0]), dim=1)
        )
        hidden = self.norm(hidden + context)
        edge_pair = torch.cat((torch.abs(hidden[edge_u] - hidden[edge_v]), hidden[edge_u] * hidden[edge_v]), dim=1)
        return (
            self.class_head(hidden),
            self.boundary_head(hidden).squeeze(1),
            torch.tanh(self.gain_head(hidden)),
            self.edge_head(edge_pair).squeeze(1),
        )


def tensors(case, device):
    result = mesh_multiclass.tensors(case, device)
    result["gain_target"] = torch.from_numpy(case["gain_target"]).to(device)
    result["gain_valid"] = torch.from_numpy(case["gain_valid"]).to(device)
    return result


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = mesh_multiclass.calculate_loss(logits, boundary_logit, values)
    valid = values["gain_valid"]
    if valid.any():
        gain_regression = F.smooth_l1_loss(gain[valid], values["gain_target"][valid])
        positive = (values["gain_target"][valid] > 0.0).float()
        sign_weight = torch.where(positive > 0.5, torch.full_like(positive, 2.5), torch.ones_like(positive))
        gain_sign = F.binary_cross_entropy_with_logits(4.0 * gain[valid], positive, weight=sign_weight)
    else:
        gain_regression = gain.sum() * 0.0
        gain_sign = gain.sum() * 0.0
    edge_target = (values["target_class"][values["edge_u"]] != values["target_class"][values["edge_v"]]).float()
    positive = edge_target.sum()
    negative = float(len(edge_target)) - positive
    positive_weight = (negative / positive.clamp(min=1.0)).clamp(min=1.0, max=8.0)
    edge_bce = F.binary_cross_entropy_with_logits(edge_logit, edge_target, pos_weight=positive_weight)
    edge_transition = edge_bce + mesh_multiclass.soft_iou_loss(torch.sigmoid(edge_logit), edge_target)
    total = base + 0.75 * gain_regression + 0.25 * gain_sign + 0.50 * edge_transition
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "gain_regression": float(gain_regression.detach().cpu()),
        "gain_sign": float(gain_sign.detach().cpu()),
        "edge_transition": float(edge_transition.detach().cpu()),
    })
    return total, row


def edge_action_gain(case, edge_probability):
    edge = case["local_edges"]
    source = np.concatenate((edge[:, 0], edge[:, 1])).astype(np.int64)
    neighbor = np.concatenate((edge[:, 1], edge[:, 0])).astype(np.int64)
    probability = np.concatenate((edge_probability, edge_probability)).astype(np.float32)
    current_instances = case["instances"][case["band_indices"]]
    old_boundary = current_instances[source] != current_instances[neighbor]
    old_expected = np.where(old_boundary, probability, 1.0 - probability)
    result = np.zeros(case["target_instances"].shape, dtype=np.float32)
    degree = np.bincount(source, minlength=len(current_instances)).astype(np.float32)
    for class_id in range(result.shape[1]):
        proposed = case["target_instances"][source, class_id]
        valid = proposed >= 0
        if not np.any(valid):
            continue
        new_boundary = proposed[valid] != current_instances[neighbor[valid]]
        new_expected = np.where(new_boundary, probability[valid], 1.0 - probability[valid])
        np.add.at(result[:, class_id], source[valid], new_expected - old_expected[valid])
    result /= np.maximum(degree[:, None], 1.0)
    return result


def predict(model, cases, device):
    model.eval()
    result = []
    with torch.no_grad():
        for case in cases:
            value = tensors(case, device)
            logits, boundary, gain, edge_logit = model(
                value["x"], value["source"], value["target"], value["edge_u"], value["edge_v"]
            )
            edge_probability = torch.sigmoid(edge_logit).cpu().numpy()
            result.append({
                "probability": torch.softmax(logits, dim=1).cpu().numpy(),
                "boundary": torch.sigmoid(boundary).cpu().numpy(),
                "gain": gain.cpu().numpy(),
                "edge_probability": edge_probability,
                "edge_gain": edge_action_gain(case, edge_probability),
            })
    return result


def apply_spec(case, prediction, spec):
    probability = prediction["probability"]
    gain = prediction["gain"]
    edge_gain = prediction["edge_gain"]
    current_class = case["current_classes"]
    target_instances = case["target_instances"]
    support_matrix = case["neighbor_support"]
    valid = target_instances >= 0
    valid[np.arange(len(valid)), current_class] = False
    choice = (
        np.log(np.clip(probability, 1.0e-8, 1.0))
        + float(spec["gain_weight"]) * gain
        + float(spec["edge_gain_weight"]) * edge_gain
        + 0.25 * support_matrix
    )
    choice[~valid] = -1.0e9
    target_class = np.argmax(choice, axis=1)
    row = np.arange(len(target_class))
    target_probability = probability[row, target_class]
    current_probability = probability[row, current_class]
    margin = target_probability - current_probability
    predicted_gain = gain[row, target_class]
    predicted_edge_gain = edge_gain[row, target_class]
    support = support_matrix[row, target_class]
    target_instance = target_instances[row, target_class]
    eligible = (
        (target_instance >= 0)
        & (target_probability >= float(spec["probability_minimum"]))
        & (margin >= float(spec["margin_minimum"]))
        & (support >= float(spec["support_minimum"]))
        & (predicted_gain >= float(spec["gain_minimum"]))
        & (predicted_edge_gain >= float(spec["edge_gain_minimum"]))
        & ~case["protected_local"]
    )
    score = (
        target_probability
        * (np.maximum(margin, 0.0) + 0.02)
        * (support + 0.10)
        * (1.0 + np.maximum(predicted_gain, 0.0))
        * (1.0 + np.maximum(predicted_edge_gain, 0.0))
    )
    current_instances = case["instances"][case["band_indices"]]
    groups = mesh_multiclass.components(
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
    maximum_instance_id = max(int(output.max()), max(mapping, default=0))
    lookup = np.zeros(maximum_instance_id + 1, dtype=np.int64)
    for instance_id, label in mapping.items():
        lookup[instance_id] = label
    return lookup[output], output, len(chosen)


def evaluate(cases, predictions, spec):
    outputs = []
    changed = []
    for case, prediction in zip(cases, predictions):
        labels, instances, count = apply_spec(case, prediction, spec)
        outputs.append((labels, instances))
        changed.append(count)
    metrics = {
        "BIoU": float(np.mean([
            mesh_multiclass.boundary_iou(case["gt_labels"], value[0], case["edges"])
            for case, value in zip(cases, outputs)
        ])),
        "IoU": float(np.mean([
            mesh_multiclass.instance_iou(case["gt_labels"], value[0])
            for case, value in zip(cases, outputs)
        ])),
    }
    return metrics, outputs, changed


def select_validation(cases, predictions):
    specs = []
                                                                                        
                                                                             
                                                                                
    for probability in (0.65, 0.80):
        for margin in (0.0,):
            for support in (0.10,):
                for cap in (0.0005, 0.001):
                    for gain_weight in (0.75,):
                        for gain_minimum in (-0.05, 0.05):
                            for edge_gain_weight in (0.25, 0.50, 1.00):
                                for edge_gain_minimum in (-0.10, 0.0):
                                    specs.append({
                                        "probability_minimum": probability,
                                        "margin_minimum": margin,
                                        "support_minimum": support,
                                        "cap_fraction": cap,
                                        "minimum_component_size": 1,
                                        "gain_weight": gain_weight,
                                        "gain_minimum": gain_minimum,
                                        "edge_gain_weight": edge_gain_weight,
                                        "edge_gain_minimum": edge_gain_minimum,
                                    })
    baseline = {
        "BIoU": float(np.mean([
            mesh_multiclass.boundary_iou(case["gt_labels"], case["labels"], case["edges"])
            for case in cases
        ])),
        "IoU": float(np.mean([
            mesh_multiclass.instance_iou(case["gt_labels"], case["labels"])
            for case in cases
        ])),
    }
    table = []
    for spec in specs:
        metrics, _, changed = evaluate(cases, predictions, spec)
        table.append({
            **spec,
            **metrics,
            "total_changed": int(sum(changed)),
            "passes_iou_guard": metrics["IoU"] >= baseline["IoU"] - 0.001,
        })
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
        rows.append({"scan_id": case["stem"], "changed": int(changed), "sha256": mesh_multiclass.sha256(path)})
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
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260866)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    started = time.time()
    train_cases = load_cases(
        args.train_pred_root, args.train_evidence_root, args.obj_root, args.json_root,
        args.band_hops, args.boundary_distance_hops, None, args.load_workers,
    )
    val_cases = load_cases(
        args.val_pred_root, args.val_evidence_root, args.obj_root, args.json_root,
        args.band_hops, args.boundary_distance_hops, args.val_protect_reference_root,
        min(args.load_workers, 4),
    )
    mean, std = mesh_multiclass.fit_standardizer(train_cases)
    mesh_multiclass.standardize(train_cases, mean, std)
    mesh_multiclass.standardize(val_cases, mean, std)
    mesh_multiclass.set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = EdgeGatedGainAwareMeshGNN(
        train_cases[0]["features"].shape[1], args.hidden_dim,
        args.graph_layers, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    order_rng = np.random.RandomState(args.seed)
    history = []
    best = None
    best_path = args.run_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for position, case_index in enumerate(order_rng.permutation(len(train_cases)), 1):
            case = train_cases[int(case_index)]
            value = tensors(case, device)
            optimizer.zero_grad()
            logits, boundary, gain, edge_logit = model(
                value["x"], value["source"], value["target"], value["edge_u"], value["edge_v"]
            )
            loss, row = calculate_loss(logits, boundary, gain, edge_logit, value)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss on {}".format(case["stem"]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(row)
            if position % 50 == 0 or position == len(train_cases):
                print("TRAIN_TRAIN epoch={} {}/{} loss={:.6f} gain_reg={:.6f} edge={:.6f}".format(
                    epoch, position, len(train_cases), row["total"], row["gain_regression"], row["edge_transition"]
                ), flush=True)
        scheduler.step()
        predictions = predict(model, val_cases, device)
        baseline, selected, table = select_validation(val_cases, predictions)
        summary = {
            "epoch": epoch,
            "loss": float(np.mean([item["total"] for item in losses])),
            "baseline": baseline,
            "selected": selected,
        }
        history.append(summary)
        print("TRAIN_EPOCH {} loss={:.6f} val_biou={:.9f} baseline={:.9f} val_iou={:.9f} spec={}".format(
            epoch, summary["loss"], selected["BIoU"], baseline["BIoU"], selected["IoU"],
            json.dumps({key: selected[key] for key in (
                "probability_minimum", "margin_minimum", "support_minimum",
                "cap_fraction", "gain_weight", "gain_minimum", "edge_gain_weight", "edge_gain_minimum",
            )}, sort_keys=True),
        ), flush=True)
        if best is None or (
            selected["BIoU"], selected["IoU"], -selected["total_changed"]
        ) > (
            best["selected"]["BIoU"], best["selected"]["IoU"],
            -best["selected"]["total_changed"],
        ):
            best = {
                "epoch": epoch,
                "selected": selected,
                "top": sorted(table, key=lambda item: (item["BIoU"], item["IoU"]), reverse=True)[:30],
            }
            torch.save({
                "schema": SCHEMA,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "input_dim": train_cases[0]["features"].shape[1],
                "hidden_dim": args.hidden_dim,
                "graph_layers": args.graph_layers,
                "dropout": args.dropout,
                "mean": mean,
                "std": std,
                "selected": selected,
            }, best_path)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict(model, val_cases, device)
    outputs = write_predictions(
        val_cases, predictions, best["selected"], args.run_dir / "predictions" / "val"
    )
    result = {
        "status": "PASS",
        "schema": SCHEMA,
        "best": best,
        "history": history,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_sha256": mesh_multiclass.sha256(best_path),
        "outputs": outputs,
        "duration_seconds": time.time() - started,
    }
    (args.run_dir / "training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("TRAIN_COMPLETE best_epoch={} val_biou={:.9f}".format(
        best["epoch"], best["selected"]["BIoU"]
    ), flush=True)


if __name__ == "__main__":
    main()



