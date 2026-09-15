# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import hashlib
import json
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
import biou_gain_ranking as biou_gain
import class_centroid_dilated as class_centroid
import boundary_benefit_gate as boundary_benefit


SCHEMA = "boundary-boundary-ownership-query-graph"
CLASS_COUNT = 17
PAIR_FEATURES = 7


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-root", type=Path)
    parser.add_argument("--train-evidence-root", type=Path)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--val-evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--mask-ratio", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261274)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-max-scans", type=int, default=0)
    return parser.parse_args()


class BoundaryOwnershipQueryGraphNet(nn.Module):
                                                                        

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
        self.assignment_head = nn.Linear(hidden_dim, CLASS_COUNT)
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
            nn.Dropout(dropout), nn.Linear(hidden_dim, CLASS_COUNT),
        )
        self.benefit_head = nn.Linear(hidden_dim, CLASS_COUNT)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

                                              
        self.point_query = nn.Linear(hidden_dim, hidden_dim)
        self.class_query = nn.Embedding(CLASS_COUNT, hidden_dim)
        self.prototype_projection = nn.Linear(hidden_dim, hidden_dim)
        self.prototype_scale = nn.Parameter(torch.tensor(-2.0))
        self.query_scale = nn.Parameter(torch.tensor(0.0))
        self.pair_bias = nn.Sequential(
            nn.Linear(PAIR_FEATURES, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    @staticmethod
    def pair_features(x, current):
                                                                                  
        semantic = x[:, 0:17]
        support = x[:, 34:51]
        start = x.shape[1] - 72
        distance_mesh = x[:, start:start + 17]
        distance_radius = x[:, start + 17:start + 34]
        fraction = x[:, start + 34:start + 51]
        present = x[:, start + 51:start + 68]
        is_current = F.one_hot(current.long(), CLASS_COUNT).to(x.dtype)
        return torch.stack((
            semantic, support, distance_mesh, distance_radius,
            fraction, present, is_current,
        ), dim=2)

    @staticmethod
    def dynamic_prototypes(hidden, current):
        global_mean = hidden.mean(dim=0)
        rows = []
        for class_id in range(CLASS_COUNT):
            selected = current == class_id
            rows.append(hidden[selected].mean(dim=0) if selected.any() else global_mean)
        return torch.stack(rows, dim=0)

    def encode(self, x, source, target):
        hidden = self.encoder(x)
        snapshots = []
        sampled = {0, len(self.blocks) // 2, len(self.blocks) - 1}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, source, target)
            if index in sampled:
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
        return self.context_norm(hidden + global_context)

    def forward(self, x, current, source, target, edge_u, edge_v):
        hidden = self.encode(x, source, target)
        prototypes = self.dynamic_prototypes(hidden, current)
        point_query = F.normalize(self.point_query(hidden), dim=1)
        prototype_query = self.class_query.weight + (
            torch.sigmoid(self.prototype_scale)
            * self.prototype_projection(prototypes)
        )
        prototype_query = F.normalize(prototype_query, dim=1)
        query_residual = torch.exp(self.query_scale.clamp(-2.0, 3.0)) * (
            point_query @ prototype_query.transpose(0, 1)
        )
        pair_bias = self.pair_bias(self.pair_features(x, current)).squeeze(2)
        assignment_logit = self.assignment_head(hidden) + query_residual + pair_bias
        edge_pair = torch.cat((
            torch.abs(hidden[edge_u] - hidden[edge_v]),
            hidden[edge_u] * hidden[edge_v],
        ), dim=1)
        return {
            "hidden": hidden,
            "assignment_logit": assignment_logit,
            "boundary_logit": self.boundary_head(hidden).squeeze(1),
            "edge_boundary_logit": self.edge_head(edge_pair).squeeze(1),
            "reconstruction": self.feature_decoder(hidden),
        }


def append_targets(case):
    boundary_benefit.append_targets(case)
    return case


def tensors(case, device):
    value = edge_gated.tensors(case, device)
    value.update({
        "candidate": torch.from_numpy(
            np.asarray(case["target_instances"], dtype=np.int64) >= 0
        ).to(device),
        "assignment_valid": torch.from_numpy(
            np.asarray(case["assignment_valid"], dtype=np.bool_)
        ).to(device),
    })
    return value


def balanced_binary(logit, target):
    each = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return boundary_tversky.balanced_mean(each, target > 0.5)


def calculate_loss(output, value, original_x):
    target = value["target_class"].long()
    current = value["current"].long()
    candidate = value["candidate"].bool()
    valid_row = value["assignment_valid"].bool()
    logits = output["assignment_logit"].masked_fill(~candidate, -1.0e4)
    probability = torch.softmax(logits, dim=1)
    row = torch.arange(len(target), device=target.device)
    target_probability = probability[row, target]
    changed = (target != current) & valid_row
    boundary_vertex = value["gt_boundary"].float()
    weight = (
        1.0 + 5.0 * changed.float() + 1.5 * boundary_vertex
    ) * valid_row.float()
    focal = torch.pow(1.0 - target_probability, 1.25)
    assignment_each = F.cross_entropy(logits, target, reduction="none")
    assignment = (assignment_each * focal * weight).sum() / weight.sum().clamp_min(1.0)

    target_score = logits[row, target]
    current_score = logits[row, current]
    migration = F.relu(0.75 - target_score + current_score)
    migration = (
        migration * changed.float()
    ).sum() / changed.float().sum().clamp_min(1.0)

    alternative = logits.clone()
    alternative[row, current] = -1.0e4
    best_alternative = alternative.max(dim=1)[0]
    stayed = (target == current) & valid_row
    stay_margin = F.relu(0.35 - current_score + best_alternative)
    stay_margin = (
        stay_margin * stayed.float()
    ).sum() / stayed.float().sum().clamp_min(1.0)

    boundary_probability = torch.sigmoid(output["boundary_logit"])
    boundary = (
        balanced_binary(output["boundary_logit"], boundary_vertex)
        + 0.75 * edge_gated.mesh_multiclass.soft_iou_loss(boundary_probability, boundary_vertex)
    )

    edge_u, edge_v = value["edge_u"], value["edge_v"]
    edge_target = (target[edge_u] != target[edge_v]).float()
    explicit_edge = (
        balanced_binary(output["edge_boundary_logit"], edge_target)
        + 0.75 * edge_gated.mesh_multiclass.soft_iou_loss(
            torch.sigmoid(output["edge_boundary_logit"]), edge_target
        )
    )
    soft_edge = (
        1.0 - (probability[edge_u] * probability[edge_v]).sum(dim=1)
    ).clamp(1.0e-5, 1.0 - 1.0e-5)
    structured_boundary = (
        F.binary_cross_entropy(soft_edge, edge_target)
        + edge_gated.mesh_multiclass.soft_iou_loss(soft_edge, edge_target)
    )

    embedding = F.normalize(output["hidden"], dim=1)
    similarity = (embedding[edge_u] * embedding[edge_v]).sum(dim=1)
    same = edge_target < 0.5
    same_loss = (1.0 - similarity[same]).mean() if same.any() else similarity.sum() * 0.0
    different = ~same
    different_loss = (
        F.relu(similarity[different] - 0.10).mean()
        if different.any() else similarity.sum() * 0.0
    )
    contrastive = same_loss + different_loss
    reconstruction = F.smooth_l1_loss(output["reconstruction"], original_x)

    total = (
        assignment + 0.80 * migration + 0.25 * stay_margin
        + 0.40 * boundary + 0.55 * explicit_edge
        + 0.75 * structured_boundary + 0.20 * contrastive
        + 0.05 * reconstruction
    )
    return total, {
        "total": float(total.detach().cpu()),
        "assignment": float(assignment.detach().cpu()),
        "migration": float(migration.detach().cpu()),
        "stay_margin": float(stay_margin.detach().cpu()),
        "boundary": float(boundary.detach().cpu()),
        "explicit_edge": float(explicit_edge.detach().cpu()),
        "structured_boundary": float(structured_boundary.detach().cpu()),
        "contrastive": float(contrastive.detach().cpu()),
        "reconstruction": float(reconstruction.detach().cpu()),
    }


def predict(model, cases, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for case in cases:
            value = tensors(case, device)
            output = model(
                value["x"], value["current"], value["source"], value["target"],
                value["edge_u"], value["edge_v"],
            )
            logits = output["assignment_logit"].masked_fill(
                ~value["candidate"].bool(), -1.0e4
            )
            rows.append({
                "logits": logits.cpu().numpy().astype(np.float32),
                "probability": torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32),
                "edge_boundary_probability": torch.sigmoid(
                    output["edge_boundary_logit"]
                ).cpu().numpy().astype(np.float32),
            })
    return rows


def remap_labels(case, output_instances):
    mapping = {}
    for instance_id in np.unique(case["instances"]):
        mask = case["instances"] == int(instance_id)
        labels, counts = np.unique(case["labels"][mask], return_counts=True)
        mapping[int(instance_id)] = int(labels[int(np.argmax(counts))])
    maximum = max(int(output_instances.max()), max(mapping, default=0))
    lookup = np.zeros(maximum + 1, dtype=np.int64)
    for instance_id, label in mapping.items():
        lookup[instance_id] = label
    return lookup[output_instances]


def component_filter(selected, current, target, edges, minimum):
    if int(minimum) <= 1 or not selected.any():
        return selected
    adjacency = [[] for _ in range(len(selected))]
    for left, right in np.asarray(edges, dtype=np.int64):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    keep = np.zeros(len(selected), dtype=bool)
    visited = np.zeros(len(selected), dtype=bool)
    for seed in np.flatnonzero(selected):
        if visited[seed]:
            continue
        signature = (int(current[seed]), int(target[seed]))
        stack, group = [int(seed)], []
        visited[seed] = True
        while stack:
            index = stack.pop()
            group.append(index)
            for neighbor in adjacency[index]:
                if (
                    selected[neighbor] and not visited[neighbor]
                    and (int(current[neighbor]), int(target[neighbor])) == signature
                ):
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(group) >= int(minimum):
            keep[np.asarray(group, dtype=np.int64)] = True
    return keep


def apply_spec(case, prediction, spec):
    if bool(spec.get("identity", False)):
        return case["labels"].copy(), case["instances"].copy(), 0
    score = np.asarray(prediction["logits"], dtype=np.float64).copy()
    candidates = np.asarray(case["target_instances"], dtype=np.int64) >= 0
    score[~candidates] = -1.0e9
    current = np.asarray(case["current_classes"], dtype=np.int64)
    edge = np.asarray(case["local_edges"], dtype=np.int64)
    boundary = np.asarray(
        prediction["edge_boundary_probability"], dtype=np.float64
    )
    graph_weight = float(spec["graph_weight"])
    labels = score.argmax(axis=1)
    refined_score = score
    if graph_weight > 0.0 and len(edge):
        same_weight = 1.0 - 2.0 * boundary
        for _ in range(int(spec.get("icm_steps", 2))):
            smooth = np.zeros_like(score)
            np.add.at(smooth, (edge[:, 0], labels[edge[:, 1]]), same_weight)
            np.add.at(smooth, (edge[:, 1], labels[edge[:, 0]]), same_weight)
            refined_score = score + graph_weight * smooth
            refined_score[~candidates] = -1.0e9
            labels = refined_score.argmax(axis=1)
    row = np.arange(len(labels), dtype=np.int64)
    margin = refined_score[row, labels] - refined_score[row, current]
    selected = (
        (labels != current)
        & (margin >= float(spec["margin_minimum"]))
        & ~np.asarray(case["protected_local"], dtype=bool)
    )
    selected = component_filter(
        selected, current, labels, edge, int(spec["minimum_component"])
    )
    chosen = np.flatnonzero(selected)
    cap_fraction = float(spec["cap_fraction"])
    if cap_fraction < 1.0:
        cap = max(1, int(round(cap_fraction * len(case["instances"]))))
        if len(chosen) > cap:
            order = np.argsort(margin[chosen], kind="mergesort")
            selected[:] = False
            selected[chosen[order[-cap:]]] = True
    target_instances = np.asarray(case["target_instances"], dtype=np.int64)
    target_instance = target_instances[row, labels]
    selected &= target_instance >= 0
    chosen = np.flatnonzero(selected)
    output_instances = np.asarray(case["instances"], dtype=np.int64).copy()
    output_instances[np.asarray(case["band_indices"], dtype=np.int64)[chosen]] = (
        target_instance[chosen]
    )
    return remap_labels(case, output_instances), output_instances, int(len(chosen))


def validation_specs(fast=True):
    rows = [{"identity": True}]
    if fast:
        graph_weights = (0.0, 0.50)
        margins = (0.0, 0.25, 0.50)
        components = (1, 3)
        caps = (0.02, 1.0)
    else:
        graph_weights = (0.0, 0.25, 0.50, 1.0)
        margins = (0.0, 0.10, 0.25, 0.50, 0.75)
        components = (1, 3, 5, 9)
        caps = (0.005, 0.01, 0.02, 0.05, 1.0)
    for graph_weight in graph_weights:
        for margin in margins:
            for minimum_component in components:
                for cap in caps:
                    rows.append({
                        "graph_weight": graph_weight,
                        "margin_minimum": margin,
                        "minimum_component": minimum_component,
                        "cap_fraction": cap,
                        "icm_steps": 2,
                    })
    return rows


def select_validation(cases, predictions, fast=True):
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
            table.append({**spec, **baseline, "total_changed": 0})
            continue
        biou, iou, changed = [], [], []
        for case, prediction, cache in zip(cases, predictions, caches):
            labels, _instances, count = apply_spec(case, prediction, spec)
            b_value, i_value = biou_gain._incremental_metrics(labels, cache)
            biou.append(b_value)
            iou.append(i_value)
            changed.append(count)
        table.append({
            **spec,
            "BIoU": float(np.mean(biou)),
            "IoU": float(np.mean(iou)),
            "total_changed": int(sum(changed)),
        })
    selected = max(table, key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed"]))
    return baseline, selected, table


def assignment_stats(cases, predictions):
    total = changed_total = correct = changed_correct = predicted_moves = true_moves = 0
    for case, prediction in zip(cases, predictions):
        candidate = np.asarray(case["target_instances"], dtype=np.int64) >= 0
        score = np.asarray(prediction["logits"], dtype=np.float64).copy()
        score[~candidate] = -1.0e9
        predicted = score.argmax(axis=1)
        target = np.asarray(case["target_classes"], dtype=np.int64)
        current = np.asarray(case["current_classes"], dtype=np.int64)
        valid = np.asarray(case["assignment_valid"], dtype=bool)
        changed = (target != current) & valid
        total += int(valid.sum())
        correct += int(np.sum((predicted == target) & valid))
        changed_total += int(changed.sum())
        changed_correct += int(np.sum((predicted == target) & changed))
        predicted_moves += int(np.sum((predicted != current) & valid))
        true_moves += int(changed.sum())
    return {
        "assignment_accuracy": float(correct / max(total, 1)),
        "changed_assignment_accuracy": float(changed_correct / max(changed_total, 1)),
        "predicted_moves": predicted_moves,
        "true_moves": true_moves,
    }


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
        rows.append({"scan_id": case["stem"], "changed": changed, "sha256": sha256(path)})
    return rows


def load_cases(args):
    edge_gated.mesh_multiclass.build_case = class_centroid.build_case
    edge_gated.add_gain_targets = biou_gain.exact_biou_gain_targets
    val_cases = edge_gated.load_cases(
        args.val_pred_root, args.val_evidence_root, args.obj_root,
        args.json_root, 3, 14, args.val_pred_root, min(args.workers, 4),
    )
    if args.smoke_max_scans > 0:
        val_cases = val_cases[:args.smoke_max_scans]
    if args.train_pred_root is None or args.train_evidence_root is None:
        train_cases = val_cases
    else:
        train_cases = edge_gated.load_cases(
            args.train_pred_root, args.train_evidence_root, args.obj_root,
            args.json_root, 3, 14, None, args.workers,
        )
        if args.smoke_max_scans > 0:
            train_cases = train_cases[:args.smoke_max_scans]
    combined_cases = list(train_cases)
    train_stems = {case["stem"] for case in train_cases}
    for case in val_cases:
        if case["stem"] in train_stems:
            continue
        combined_cases.append(case)
        train_stems.add(case["stem"])
    for case in combined_cases:
        append_targets(case)
    return train_cases, val_cases


def main():
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    train_cases, val_cases = load_cases(args)
    warmstart = torch.load(args.warmstart, map_location="cpu")
    mean = np.asarray(warmstart["mean"], dtype=np.float32)
    std = np.asarray(warmstart["std"], dtype=np.float32)
    edge_gated.mesh_multiclass.standardize(train_cases, mean, std)
    if val_cases is not train_cases:
        edge_gated.mesh_multiclass.standardize(val_cases, mean, std)

    device = torch.device(args.device)
    model = BoundaryOwnershipQueryGraphNet(
        len(mean), args.hidden_dim, args.layers, args.dropout
    ).to(device)
    missing, unexpected = model.load_state_dict(warmstart["state_dict"], strict=False)
    allowed_missing = (
        "point_query.", "class_query.", "prototype_projection.",
        "prototype_scale", "query_scale", "pair_bias.", "feature_decoder.",
    )
    bad_missing = [name for name in missing if not name.startswith(allowed_missing)]
    if bad_missing or unexpected:
        raise RuntimeError(f"warmstart mismatch missing={bad_missing} unexpected={unexpected}")
    with torch.no_grad():
        if model.point_query.weight.shape[0] == model.point_query.weight.shape[1]:
            model.point_query.weight.copy_(torch.eye(model.point_query.weight.shape[0], device=device))
            model.point_query.bias.zero_()
        model.class_query.weight.copy_(model.assignment_head.weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05,
    )
    history, best = [], None
    best_path = args.run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_rows = []
        order = np.random.RandomState(args.seed + epoch).permutation(len(train_cases))
        for position, case_index in enumerate(order, 1):
            value = tensors(train_cases[int(case_index)], device)
            original_x = value["x"]
            if args.mask_ratio > 0.0:
                mask = torch.rand_like(original_x) < float(args.mask_ratio)
                model_x = original_x.masked_fill(mask, 0.0)
            else:
                model_x = original_x
            optimizer.zero_grad(set_to_none=True)
            output = model(
                model_x, value["current"], value["source"], value["target"],
                value["edge_u"], value["edge_v"],
            )
            loss, row = calculate_loss(output, value, original_x)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            row["gradient_norm"] = float(gradient.detach().cpu())
            loss_rows.append(row)
            if position % 4 == 0 or position == len(order):
                print(
                    "TRAIN_TRAIN epoch={} {}/{} loss={:.6f} assign={:.6f} "
                    "migration={:.6f} edge={:.6f} grad={:.6f}".format(
                        epoch, position, len(order), row["total"], row["assignment"],
                        row["migration"], row["structured_boundary"], row["gradient_norm"],
                    ), flush=True,
                )
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "loss": float(np.mean([row["total"] for row in loss_rows])),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            predictions = predict(model, val_cases, device)
            baseline, selected, table = select_validation(val_cases, predictions, fast=True)
            epoch_row.update({
                "baseline": baseline,
                "selected": selected,
                "assignment": assignment_stats(val_cases, predictions),
            })
            key = (selected["BIoU"], selected["IoU"], -selected["total_changed"])
            if best is None or key > tuple(best["key"]):
                best = {
                    "key": list(key), "epoch": epoch, "selected": selected,
                    "assignment": epoch_row["assignment"],
                    "top": sorted(
                        table, key=lambda item: (
                            item["BIoU"], item["IoU"], -item["total_changed"]
                        ), reverse=True,
                    )[:20],
                }
                torch.save({
                    "schema": SCHEMA, "epoch": epoch,
                    "state_dict": model.state_dict(), "input_dim": len(mean),
                    "hidden_dim": args.hidden_dim, "layers": args.layers,
                    "dropout": args.dropout, "mean": mean, "std": std,
                    "selected": selected, "warmstart": str(args.warmstart),
                }, best_path)
            print(
                "TRAIN_EPOCH {} loss={:.6f} BIoU={:.9f} baseline={:.9f} "
                "changed={} assign={:.5f} changed_assign={:.5f}".format(
                    epoch, epoch_row["loss"], selected["BIoU"], baseline["BIoU"],
                    selected["total_changed"], epoch_row["assignment"]["assignment_accuracy"],
                    epoch_row["assignment"]["changed_assignment_accuracy"],
                ), flush=True,
            )
        history.append(epoch_row)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = predict(model, val_cases, device)
    baseline, selected, table = select_validation(val_cases, predictions, fast=False)
    outputs = write_predictions(
        val_cases, predictions, selected, args.run_dir / "predictions" / "val"
    )
    result = {
        "schema": SCHEMA,
        "status": "VALIDATION",
        "baseline": baseline,
        "selected": selected,
        "improvement": {
            "BIoU": float(selected["BIoU"] - baseline["BIoU"]),
            "IoU": float(selected["IoU"] - baseline["IoU"]),
        },
        "best_epoch": int(best["epoch"]),
        "assignment": assignment_stats(val_cases, predictions),
        "top20": sorted(
            table, key=lambda item: (item["BIoU"], item["IoU"]), reverse=True
        )[:20],
        "history": history,
        "outputs": outputs,
        "train_scans": len(train_cases),
        "val_scans": len(val_cases),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "warmstart": str(args.warmstart),
        "warmstart_sha256": sha256(args.warmstart),
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256(best_path),
        "duration_seconds": time.time() - started,
    }
    result_name = "TRAIN_RESULT.json" if SCHEMA.startswith("boundary") else "TRAIN_RESULT.json"
    result_path = args.run_dir / result_name
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "TRAIN_COMPLETE epoch={} BIoU={:.9f} baseline={:.9f} delta={:+.9f} changed={}".format(
            result["best_epoch"], selected["BIoU"], baseline["BIoU"],
            result["improvement"]["BIoU"], selected["total_changed"],
        ), flush=True,
    )


if __name__ == "__main__":
    main()


