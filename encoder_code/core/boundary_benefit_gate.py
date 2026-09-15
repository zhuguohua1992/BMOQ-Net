# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import mesh_attention_gnn as mesh_attention
import boundary_tversky_attention as boundary_tversky
import multiview_guided_gcbl as multiview_guided
import biou_gain_ranking as biou_gain
import class_centroid_dilated as class_centroid


SCHEMA = "boundary-boundary-displacement-instance-assignment"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BoundaryDisplacementAssignmentNet(nn.Module):
                                                                             

    def __init__(self, input_dim, hidden_dim=96, layers=5, dropout=0.08):
        super().__init__()
        layers = max(int(layers), 3)
        self.encoder = edge_gated.mesh_multiclass.MLP(input_dim, hidden_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList([
            mesh_attention.MeshAttentionBlock(hidden_dim, 4, dropout)
            for _ in range(layers)
        ])
        self.scale_fusion = edge_gated.mesh_multiclass.MLP(
            hidden_dim * 3, hidden_dim, hidden_dim, dropout
        )
        self.boundary_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.boundary_residual = edge_gated.mesh_multiclass.MLP(
            hidden_dim * 3, hidden_dim, hidden_dim, dropout
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.context = edge_gated.mesh_multiclass.MLP(
            hidden_dim * 2, hidden_dim, hidden_dim, dropout
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.assignment_head = nn.Linear(hidden_dim, 17)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 17),
        )
        self.benefit_head = nn.Linear(hidden_dim, 17)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, source, target, edge_u, edge_v):
        hidden = self.encoder(x)
        snapshots = []
        sample_layers = {0, len(self.blocks) // 2, len(self.blocks) - 1}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, source, target)
            if index in sample_layers:
                snapshots.append(hidden)
        multi_scale = torch.cat(snapshots, dim=1)
        base = self.scale_fusion(multi_scale)
        gate = self.boundary_gate(multi_scale)
        residual = self.boundary_residual(multi_scale)
        hidden = self.fusion_norm(base + gate * residual)
        global_context = self.context(torch.cat((
            hidden.mean(dim=0, keepdim=True),
            hidden.max(dim=0, keepdim=True)[0],
        ), dim=1))
        hidden = self.context_norm(hidden + global_context)
        edge_pair = torch.cat((
            torch.abs(hidden[edge_u] - hidden[edge_v]),
            hidden[edge_u] * hidden[edge_v],
        ), dim=1)
        return {
            "assignment_logit": self.assignment_head(hidden),
            "boundary_logit": self.boundary_head(hidden).squeeze(1),
            "distance": torch.sigmoid(self.distance_head(hidden).squeeze(1)),
            "move_logit": self.move_head(hidden).squeeze(1),
            "utility": torch.tanh(self.utility_head(hidden)),
            "benefit_logit": self.benefit_head(hidden),
            "edge_logit": self.edge_head(edge_pair).squeeze(1),
        }


def append_targets(case, maximum_hops=14):
    boundary = edge_gated.mesh_multiclass.boundary_vertices(case["gt_labels"], case["edges"])
    distance = edge_gated.mesh_multiclass.graph_distance(
        boundary, case["edges"], int(maximum_hops)
    )
    case["gt_distance_local"] = np.clip(
        distance[case["band_indices"]] / float(maximum_hops), 0.0, 1.0
    ).astype(np.float32)
    candidate = np.asarray(case["target_instances"], dtype=np.int64) >= 0
    target = np.asarray(case["target_classes"], dtype=np.int64)
    case["assignment_valid"] = candidate[
        np.arange(len(target), dtype=np.int64), target
    ]
    return case


def tensors(case, device):
    value = edge_gated.tensors(case, device)
    value.update({
        "distance": torch.from_numpy(case["gt_distance_local"]).to(device),
        "candidate": torch.from_numpy(
            np.asarray(case["target_instances"]) >= 0
        ).to(device),
        "assignment_valid": torch.from_numpy(case["assignment_valid"]).to(device),
        "gain_target": torch.from_numpy(case["gain_target"]).to(device),
        "gain_valid": torch.from_numpy(case["gain_valid"]).to(device),
    })
    return value


def balanced_binary(logit, target):
    each = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return boundary_tversky.balanced_mean(each, target > 0.5)


def calculate_loss(output, value):
    target = value["target_class"].long()
    current = value["current"].long()
    valid_row = value["assignment_valid"].bool()
    candidate = value["candidate"].bool()
    masked_logit = output["assignment_logit"].masked_fill(~candidate, -1.0e4)
    log_probability = F.log_softmax(masked_logit, dim=1)
    probability = torch.softmax(masked_logit, dim=1)
    target_probability = probability.gather(1, target[:, None]).squeeze(1)
    assignment_each = -log_probability.gather(1, target[:, None]).squeeze(1)
    changed = (target != current) & valid_row
    assignment_weight = torch.where(
        changed, torch.full_like(assignment_each, 8.0),
        torch.ones_like(assignment_each),
    )
    assignment_weight = assignment_weight * valid_row.float()
    focal = torch.pow(1.0 - target_probability, 1.5)
    assignment = (
        assignment_each * focal * assignment_weight
    ).sum() / assignment_weight.sum().clamp(min=1.0)

    move_target = changed.float()
    move = balanced_binary(output["move_logit"], move_target)

    boundary_target = value["gt_boundary"].float()
    boundary_probability = torch.sigmoid(output["boundary_logit"])
    boundary_bce = balanced_binary(output["boundary_logit"], boundary_target)
    boundary_tversky = boundary_tversky.soft_tversky(
        boundary_probability, boundary_target, alpha=0.45, beta=0.55
    )
    boundary_iou = edge_gated.mesh_multiclass.soft_iou_loss(
        boundary_probability, boundary_target
    )
    distance = F.smooth_l1_loss(
        output["distance"], value["distance"].float(), beta=0.10
    )
    boundary_field = (
        boundary_bce + 0.70 * boundary_tversky
        + 0.70 * boundary_iou + 0.55 * distance
    )

    utility_valid = value["gain_valid"].bool()
    if utility_valid.any():
        utility_regression = F.smooth_l1_loss(
            output["utility"][utility_valid],
            value["gain_target"].float()[utility_valid], beta=0.05,
        )
    else:
        utility_regression = output["utility"].sum() * 0.0
    listwise, utility_sign, pairwise = biou_gain._exact_ranking_losses(
        output["utility"], value
    )
    if utility_valid.any():
        benefit_target = (value["gain_target"] > 0.0).float()
        benefit_logit = output["benefit_logit"][utility_valid]
        benefit_truth = benefit_target[utility_valid]
        positive = benefit_truth.sum()
        negative = benefit_truth.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
        benefit = F.binary_cross_entropy_with_logits(
            benefit_logit, benefit_truth, pos_weight=pos_weight
        )
    else:
        benefit = output["benefit_logit"].sum() * 0.0

    edge_u, edge_v = value["edge_u"], value["edge_v"]
    edge_target = (target[edge_u] != target[edge_v]).float()
    edge_explicit = balanced_binary(output["edge_logit"], edge_target)
    edge_from_assignment = (
        1.0 - (probability[edge_u] * probability[edge_v]).sum(dim=1)
    ).clamp(1.0e-5, 1.0 - 1.0e-5)
    edge_consistency = (
        F.binary_cross_entropy(edge_from_assignment, edge_target)
        + edge_gated.mesh_multiclass.soft_iou_loss(edge_from_assignment, edge_target)
    )

    total = (
        assignment + 0.45 * move + 0.55 * boundary_field
        + 0.70 * utility_regression + 0.20 * listwise
        + 0.25 * utility_sign + 0.35 * pairwise
        + 1.00 * benefit
        + 0.30 * edge_explicit + 0.45 * edge_consistency
    )
    return total, {
        "total": float(total.detach().cpu()),
        "assignment": float(assignment.detach().cpu()),
        "move": float(move.detach().cpu()),
        "boundary_field": float(boundary_field.detach().cpu()),
        "distance": float(distance.detach().cpu()),
        "utility_regression": float(utility_regression.detach().cpu()),
        "benefit": float(benefit.detach().cpu()),
        "utility_listwise": float(listwise.detach().cpu()),
        "utility_sign": float(utility_sign.detach().cpu()),
        "utility_pairwise": float(pairwise.detach().cpu()),
        "edge_explicit": float(edge_explicit.detach().cpu()),
        "edge_consistency": float(edge_consistency.detach().cpu()),
        "changed_fraction": float(changed.float().mean().detach().cpu()),
        "assignable_fraction": float(valid_row.float().mean().detach().cpu()),
    }


def predict(model, cases, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for case in cases:
            value = tensors(case, device)
            output = model(
                value["x"], value["source"], value["target"],
                value["edge_u"], value["edge_v"],
            )
            candidate = value["candidate"].bool()
            masked_logit = output["assignment_logit"].masked_fill(
                ~candidate, -1.0e4
            )
            boundary = torch.sigmoid(output["boundary_logit"])
            distance = output["distance"]
            rows.append({
                "probability": torch.softmax(masked_logit, dim=1).cpu().numpy(),
                "move": torch.sigmoid(output["move_logit"]).cpu().numpy(),
                "boundary": boundary.cpu().numpy(),
                "distance": distance.cpu().numpy(),
                "displacement": (
                    torch.sigmoid(output["move_logit"])
                    * (0.65 * boundary + 0.35 * (1.0 - distance))
                ).cpu().numpy(),
                "utility": torch.sigmoid(
                    output["benefit_logit"]
                ).cpu().numpy(),
            })
    return rows


def apply_spec(case, prediction, spec):
    if bool(spec.get("identity", False)):
        return case["labels"].copy(), case["instances"].copy(), 0
    probability = np.asarray(prediction["probability"], dtype=np.float32)
    utility = np.asarray(prediction["utility"], dtype=np.float32)
    current = np.asarray(case["current_classes"], dtype=np.int64)
    targets = np.asarray(case["target_instances"], dtype=np.int64)
    support = np.asarray(case["neighbor_support"], dtype=np.float32)
    valid = targets >= 0
    valid[np.arange(len(valid)), current] = False
    choice = (
        float(spec["assignment_weight"])
        * np.log(np.clip(probability, 1.0e-8, 1.0))
        + float(spec["utility_weight"]) * utility
        + 0.10 * support
    )
    choice[~valid] = -1.0e9
    target_class = np.argmax(choice, axis=1)
    row = np.arange(len(target_class))
    target_instance = targets[row, target_class]
    target_probability = probability[row, target_class]
    current_probability = probability[row, current]
    margin = target_probability - current_probability
    target_utility = utility[row, target_class]
    target_support = support[row, target_class]
    displacement = np.asarray(prediction["displacement"], dtype=np.float32)
    eligible = (
        (target_instance >= 0)
        & (target_probability >= float(spec["probability_minimum"]))
        & (margin >= float(spec["margin_minimum"]))
        & (target_utility >= float(spec["utility_minimum"]))
        & (displacement >= float(spec["displacement_minimum"]))
        & ~np.asarray(case["protected_local"], dtype=bool)
    )
    score = (
        float(spec["utility_weight"]) * target_utility
        + float(spec["assignment_weight"])
        * np.log(np.clip(target_probability, 1.0e-8, 1.0))
        + 0.10 * target_support + 0.05 * displacement + 0.01 * margin
    )
    chosen = np.flatnonzero(eligible)
    cap = max(1, int(round(
        float(spec["cap_fraction"]) * len(case["instances"])
    )))
    if len(chosen) > cap:
        order = np.argsort(score[chosen], kind="mergesort")
        chosen = chosen[order[-cap:]]

    output_instances = np.asarray(case["instances"], dtype=np.int64).copy()
    output_instances[np.asarray(case["band_indices"])[chosen]] = target_instance[chosen]
    mapping = {}
    for instance_id in np.unique(case["instances"]):
        mask = case["instances"] == int(instance_id)
        labels, counts = np.unique(case["labels"][mask], return_counts=True)
        mapping[int(instance_id)] = int(labels[int(np.argmax(counts))])
    lookup = np.zeros(max(int(output_instances.max()), max(mapping, default=0)) + 1,
                      dtype=np.int64)
    for instance_id, label in mapping.items():
        lookup[instance_id] = label
    return lookup[output_instances], output_instances, int(len(chosen))


def validation_specs(fast=False):
    rows = [{"identity": True}]
    if fast:
        assignment_weights = (0.5, 1.0)
        utility_weights = (1.0, 2.0)
        probability_minimums = (0.0, 0.15)
        displacement_minimums = (0.20, 0.45)
        utility_minimums = (0.0, 0.05)
        caps = (0.0005, 0.001, 0.002, 0.005)
    else:
        assignment_weights = (0.0, 0.5, 1.0)
        utility_weights = (1.0, 2.0, 4.0)
        probability_minimums = (0.0, 0.10, 0.25)
        displacement_minimums = (0.15, 0.35)
        utility_minimums = (-0.05, 0.0, 0.05, 0.10)
        caps = (0.00025, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01)
    for assignment_weight in assignment_weights:
        for utility_weight in utility_weights:
            for probability_minimum in probability_minimums:
                for displacement_minimum in displacement_minimums:
                    for utility_minimum in utility_minimums:
                        for cap in caps:
                            rows.append({
                                "assignment_weight": assignment_weight,
                                "utility_weight": utility_weight,
                                "probability_minimum": probability_minimum,
                                "margin_minimum": -1.0,
                                "displacement_minimum": displacement_minimum,
                                "utility_minimum": utility_minimum,
                                "cap_fraction": cap,
                            })
    return rows


def select_validation(cases, predictions, fast=False):
    caches = [biou_gain._metric_cache(case) for case in cases]
    baseline = {
        "BIoU": float(np.mean([
            biou_gain._incremental_metrics(case["labels"], cache)[0]
            for case, cache in zip(cases, caches)
        ])),
        "IoU": float(np.mean([
            biou_gain._incremental_metrics(case["labels"], cache)[1]
            for case, cache in zip(cases, caches)
        ])),
    }
    table = []
    for spec in validation_specs(fast=fast):
        if spec.get("identity"):
            table.append({
                **spec, **baseline, "total_changed": 0,
                "passes_iou_guard": True,
            })
            continue
        biou_rows, iou_rows, changed_rows = [], [], []
        for case, prediction, cache in zip(cases, predictions, caches):
            labels, _instances, changed = apply_spec(case, prediction, spec)
            biou, iou = biou_gain._incremental_metrics(labels, cache)
            biou_rows.append(biou)
            iou_rows.append(iou)
            changed_rows.append(changed)
        row = {
            **spec,
            "BIoU": float(np.mean(biou_rows)),
            "IoU": float(np.mean(iou_rows)),
            "total_changed": int(sum(changed_rows)),
        }
        row["passes_iou_guard"] = row["IoU"] >= baseline["IoU"] - 0.001
        table.append(row)
    selected = max(
        (row for row in table if row["passes_iou_guard"]),
        key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed"]),
    )
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
        rows.append({
            "scan_id": case["stem"], "changed": int(changed),
            "sha256": sha256(path),
        })
    return rows


def subset_root(source, destination, maximum):
    if int(maximum) <= 0:
        return source
    destination.mkdir(parents=True, exist_ok=False)
    paths = sorted(source.glob("*.json"))[:int(maximum)]
    if not paths:
        raise RuntimeError(f"no JSON predictions in {source}")
    for path in paths:
        os.symlink(path, destination / path.name)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-root", type=Path, required=True)
    parser.add_argument("--train-evidence-root", type=Path, required=True)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--val-evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20261259)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-max-scans", type=int, default=0)
    parser.add_argument("--val-max-scans", type=int, default=0)
    parser.add_argument("--fast-selection", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--benefit-only", action="store_true")
    args = parser.parse_args()

    if args.resume is None:
        if args.run_dir.exists():
            raise FileExistsError(args.run_dir)
        args.run_dir.mkdir(parents=True)
    elif not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_root = subset_root(
        args.train_pred_root, args.run_dir / "subset_train",
        args.train_max_scans,
    )
    val_root = subset_root(
        args.val_pred_root, args.run_dir / "subset_val", args.val_max_scans,
    )
    edge_gated.mesh_multiclass.build_case = class_centroid.build_case
    edge_gated.add_gain_targets = biou_gain.exact_biou_gain_targets
    train_cases = edge_gated.load_cases(
        train_root, args.train_evidence_root, args.obj_root, args.json_root,
        3, 14, None, args.workers,
    )
    val_cases = edge_gated.load_cases(
        val_root, args.val_evidence_root, args.obj_root, args.json_root,
        3, 14, val_root, min(args.workers, 4),
    )
    for case in train_cases + val_cases:
        append_targets(case)
    mean, std = edge_gated.mesh_multiclass.fit_standardizer(train_cases)
    edge_gated.mesh_multiclass.standardize(train_cases, mean, std)
    edge_gated.mesh_multiclass.standardize(val_cases, mean, std)

    device = torch.device(args.device)
    model = BoundaryDisplacementAssignmentNet(
        len(mean), args.hidden_dim, args.layers, args.dropout
    ).to(device)
    if args.warmstart is not None:
        source = torch.load(args.warmstart, map_location=device)
        incompatibility = model.load_state_dict(
            source["state_dict"], strict=False
        )
        if set(incompatibility.missing_keys) != {
            "benefit_head.weight", "benefit_head.bias"
        } or incompatibility.unexpected_keys:
            raise RuntimeError(
                "warmstart mismatch missing={} unexpected={}".format(
                    incompatibility.missing_keys,
                    incompatibility.unexpected_keys,
                )
            )
    if args.benefit_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("benefit_head."))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    history, best, start_epoch = [], None, 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best = checkpoint["best"]
        start_epoch = int(checkpoint["epoch"]) + 1

    best_path = args.run_dir / "best_model.pt"
    latest_path = args.run_dir / "latest_model.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_rows = []
        order = np.random.RandomState(args.seed + epoch).permutation(
            len(train_cases)
        )
        for position, index in enumerate(order, 1):
            value = tensors(train_cases[int(index)], device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                value["x"], value["source"], value["target"],
                value["edge_u"], value["edge_v"],
            )
            loss, loss_row = calculate_loss(output, value)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_row["gradient_norm"] = float(gradient_norm.detach().cpu())
            loss_rows.append(loss_row)
            if position % 25 == 0 or position == len(order):
                print(
                    "TRAIN_TRAIN epoch={} {}/{} loss={:.6f} assign={:.6f} "
                    "boundary={:.6f} utility={:.6f} grad={:.6f}".format(
                        epoch, position, len(order), loss_row["total"],
                        loss_row["assignment"], loss_row["boundary_field"],
                        loss_row["utility_regression"], loss_row["gradient_norm"],
                    ), flush=True,
                )
        scheduler.step()
        predictions = predict(model, val_cases, device)
        baseline, selected, table = select_validation(
            val_cases, predictions,
            fast=args.fast_selection or args.val_max_scans > 0,
        )
        epoch_row = {
            "epoch": epoch,
            "loss": float(np.mean([row["total"] for row in loss_rows])),
            "baseline": baseline,
            "selected": selected,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(
            "TRAIN_EPOCH {} loss={:.6f} val_biou={:.9f} baseline={:.9f} "
            "val_iou={:.9f} changed={}".format(
                epoch, epoch_row["loss"], selected["BIoU"],
                baseline["BIoU"], selected["IoU"], selected["total_changed"],
            ), flush=True,
        )
        key = (selected["BIoU"], selected["IoU"], -selected["total_changed"])
        if best is None or key > tuple(best["key"]):
            best = {
                "key": list(key), "epoch": epoch, "selected": selected,
                "top": sorted(
                    table,
                    key=lambda row: (
                        row["BIoU"], row["IoU"], -row["total_changed"]
                    ), reverse=True,
                )[:20],
            }
            torch.save({
                "schema": SCHEMA, "epoch": epoch,
                "state_dict": model.state_dict(), "input_dim": len(mean),
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "dropout": args.dropout, "mean": mean, "std": std,
                "selected": selected,
            }, best_path)
        torch.save({
            "schema": SCHEMA, "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history, "best": best,
        }, latest_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict(model, val_cases, device)
    outputs = write_predictions(
        val_cases, predictions, best["selected"],
        args.run_dir / "predictions" / "val",
    )
    result = {
        "schema": SCHEMA, "status": "PASS", "history": history,
        "best": {key: value for key, value in best.items() if key != "key"},
        "outputs": outputs, "duration_seconds": time.time() - started,
        "checkpoint_sha256": sha256(best_path),
        "train_scans": len(train_cases), "val_scans": len(val_cases),
    }
    result_path = args.run_dir / "benefit_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "TRAIN_COMPLETE epoch={} BIoU={:.9f} baseline={:.9f} changed={}".format(
            best["epoch"], best["selected"]["BIoU"],
            history[0]["baseline"]["BIoU"],
            best["selected"]["total_changed"],
        ), flush=True,
    )


if __name__ == "__main__":
    main()


