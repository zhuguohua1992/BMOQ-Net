# 本文件用于实现完整模型的边界编码、训练或推理。
import os

import numpy as np
import torch
import torch.nn.functional as F

import boundary_query_graph as query_graph


SCHEMA = "boundary-boundary-ownership-assignment-gate"
CLASS_COUNT = query_graph.CLASS_COUNT
BASE_TENSORS = query_graph.tensors
BASE_ASSIGNMENT_STATS = query_graph.assignment_stats


class BoundaryOwnershipAssignmentGateNet(query_graph.BoundaryOwnershipQueryGraphNet):
    def forward(self, x, current, source, target, edge_u, edge_v):
        output = super().forward(x, current, source, target, edge_u, edge_v)
        hidden = output["hidden"]
        pair_bias = self.pair_bias(self.pair_features(x, current)).squeeze(2)
        output["move_logit"] = self.move_head(hidden).squeeze(1)
        output["benefit_logit"] = self.benefit_head(hidden) + 0.5 * pair_bias
        return output


def tensors(case, device):
    value = BASE_TENSORS(case, device)
    value.update({
        "gain_target": torch.from_numpy(
            np.asarray(case["gain_target"], dtype=np.float32)
        ).to(device),
        "gain_valid": torch.from_numpy(
            np.asarray(case["gain_valid"], dtype=np.bool_)
        ).to(device),
    })
    return value


def balanced_binary(logit, target, valid):
    if not valid.any():
        return logit.sum() * 0.0
    each = F.binary_cross_entropy_with_logits(
        logit[valid], target[valid], reduction="none"
    )
    truth = target[valid] > 0.5
    positive = each[truth].mean() if truth.any() else each.sum() * 0.0
    negative = each[~truth].mean() if (~truth).any() else each.sum() * 0.0
    return 0.5 * (positive + negative)


def hard_negative_binary(logit, target, valid, negative_ratio=4):
    if not valid.any():
        return logit.sum() * 0.0
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    selected = positive.clone()
    negative_index = torch.nonzero(negative, as_tuple=False)
    positive_count = int(positive.sum().detach().cpu())
    keep_count = min(
        len(negative_index), max(1024, int(negative_ratio) * max(positive_count, 1))
    )
    if keep_count:
        negative_score = logit[negative]
        order = torch.topk(negative_score, keep_count, sorted=False).indices
        chosen = negative_index[order]
        selected[tuple(chosen.transpose(0, 1))] = True
    return F.binary_cross_entropy_with_logits(logit[selected], target[selected])


def calculate_loss(output, value, original_x):
    target = value["target_class"].long()
    current = value["current"].long()
    candidate = value["candidate"].bool()
    valid_row = value["assignment_valid"].bool()
    logits = output["assignment_logit"].masked_fill(~candidate, -1.0e4)
    probability = torch.softmax(logits, dim=1)
    row_index = torch.arange(len(target), device=target.device)
    target_probability = probability[row_index, target]
    changed = (target != current) & valid_row

    assignment_each = F.cross_entropy(logits, target, reduction="none")
    assignment_weight = (
        1.0 + 2.0 * changed.float() + 1.0 * value["gt_boundary"].float()
    ) * valid_row.float()
    assignment_focal = torch.pow(
        (1.0 - target_probability).clamp_min(1.0e-4), 0.75
    )
    assignment = (
        assignment_each * assignment_focal * assignment_weight
    ).sum() / assignment_weight.sum().clamp_min(1.0)

    target_score = logits[row_index, target]
    current_score = logits[row_index, current]
    migration = F.relu(0.75 - target_score + current_score)
    migration = (migration * changed.float()).sum() / changed.sum().clamp_min(1)

    alternative = logits.clone()
    alternative[row_index, current] = -1.0e4
    best_alternative = alternative.max(dim=1)[0]
    stayed = (target == current) & valid_row
    stay_margin = F.relu(1.0 - current_score + best_alternative)
    stay_margin = (stay_margin * stayed.float()).sum() / stayed.sum().clamp_min(1)

    move_target = changed.float()
    move_logit = output["move_logit"]
    move = (
        0.65 * F.binary_cross_entropy_with_logits(
            move_logit[valid_row], move_target[valid_row]
        )
        + 0.35 * balanced_binary(move_logit, move_target, valid_row)
    )
    positive_move_margin = F.relu(1.0 - move_logit[changed]).mean() \
        if changed.any() else move_logit.sum() * 0.0
    negative_move_margin = F.relu(1.0 + move_logit[stayed]).mean() \
        if stayed.any() else move_logit.sum() * 0.0
    move_precision = move + 0.35 * positive_move_margin + 0.75 * negative_move_margin

    gain_valid = value["gain_valid"].bool() & candidate
    gain_valid[row_index, current] = False
    benefit_target = (value["gain_target"] > 0.0).float()
    benefit_logit = output["benefit_logit"]
    benefit = (
        0.45 * F.binary_cross_entropy_with_logits(
            benefit_logit[gain_valid], benefit_target[gain_valid]
        )
        + 0.55 * hard_negative_binary(
            benefit_logit, benefit_target, gain_valid, negative_ratio=5
        )
    ) if gain_valid.any() else benefit_logit.sum() * 0.0

    positive_action = gain_valid & (benefit_target > 0.5)
    negative_action = gain_valid & ~positive_action
    has_both = positive_action.any(dim=1) & negative_action.any(dim=1)
    if has_both.any():
        positive_best = benefit_logit.masked_fill(
            ~positive_action, -1.0e4
        ).max(dim=1)[0]
        negative_best = benefit_logit.masked_fill(
            ~negative_action, -1.0e4
        ).max(dim=1)[0]
        benefit_ranking = F.relu(
            1.0 - positive_best[has_both] + negative_best[has_both]
        ).mean()
    else:
        benefit_ranking = benefit_logit.sum() * 0.0

    boundary_target = value["gt_boundary"].float()
    boundary_probability = torch.sigmoid(output["boundary_logit"])
    boundary = (
        balanced_binary(
            output["boundary_logit"], boundary_target,
            torch.ones_like(boundary_target, dtype=torch.bool),
        )
        + 0.5 * query_graph.edge_gated.mesh_multiclass.soft_iou_loss(
            boundary_probability, boundary_target
        )
    )

    edge_u, edge_v = value["edge_u"], value["edge_v"]
    edge_target = (target[edge_u] != target[edge_v]).float()
    edge_valid = torch.ones_like(edge_target, dtype=torch.bool)
    explicit_edge = (
        balanced_binary(output["edge_boundary_logit"], edge_target, edge_valid)
        + 0.5 * query_graph.edge_gated.mesh_multiclass.soft_iou_loss(
            torch.sigmoid(output["edge_boundary_logit"]), edge_target
        )
    )
    soft_edge = (
        1.0 - (probability[edge_u] * probability[edge_v]).sum(dim=1)
    ).clamp(1.0e-5, 1.0 - 1.0e-5)
    structured_boundary = (
        F.binary_cross_entropy(soft_edge, edge_target)
        + query_graph.edge_gated.mesh_multiclass.soft_iou_loss(soft_edge, edge_target)
    )

    embedding = F.normalize(output["hidden"], dim=1)
    similarity = (embedding[edge_u] * embedding[edge_v]).sum(dim=1)
    same = edge_target < 0.5
    contrastive = (
        (1.0 - similarity[same]).mean()
        + F.relu(similarity[~same] - 0.10).mean()
    )
    reconstruction = F.smooth_l1_loss(output["reconstruction"], original_x)

    total = (
        assignment + 0.70 * migration + 1.50 * stay_margin
        + 0.90 * move_precision + 1.20 * benefit
        + 0.50 * benefit_ranking + 0.20 * boundary
        + 0.25 * explicit_edge + 0.30 * structured_boundary
        + 0.10 * contrastive + 0.02 * reconstruction
    )
    return total, {
        "total": float(total.detach().cpu()),
        "assignment": float(assignment.detach().cpu()),
        "migration": float(migration.detach().cpu()),
        "stay_margin": float(stay_margin.detach().cpu()),
        "move_precision": float(move_precision.detach().cpu()),
        "benefit": float(benefit.detach().cpu()),
        "benefit_ranking": float(benefit_ranking.detach().cpu()),
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
                "move_probability": torch.sigmoid(
                    output["move_logit"]
                ).cpu().numpy().astype(np.float32),
                "benefit_logit": output[
                    "benefit_logit"
                ].cpu().numpy().astype(np.float32),
                "benefit_probability": torch.sigmoid(
                    output["benefit_logit"]
                ).cpu().numpy().astype(np.float32),
                "edge_boundary_probability": torch.sigmoid(
                    output["edge_boundary_logit"]
                ).cpu().numpy().astype(np.float32),
            })
    return rows


def validation_specs(fast=True):
    rows = [{"identity": True}]
    if fast:
        move_values = (0.40, 0.60, 0.80)
        benefit_values = (0.40, 0.60, 0.80)
        margins = (0.0, 0.50)
        caps = (0.005, 0.02, 0.05)
        components = (1,)
        benefit_weights = (0.50,)
    else:
        move_values = (0.30, 0.50, 0.70, 0.85)
        benefit_values = (0.30, 0.50, 0.70, 0.85)
        margins = (0.0, 0.25, 0.50)
        caps = (0.005, 0.01, 0.02, 0.05)
        components = (1,)
        benefit_weights = (0.50,)
    for move_minimum in move_values:
        for benefit_minimum in benefit_values:
            for margin_minimum in margins:
                for cap_fraction in caps:
                    for minimum_component in components:
                        for benefit_weight in benefit_weights:
                            rows.append({
                                "move_minimum": move_minimum,
                                "benefit_minimum": benefit_minimum,
                                "margin_minimum": margin_minimum,
                                "cap_fraction": cap_fraction,
                                "minimum_component": minimum_component,
                                "benefit_weight": benefit_weight,
                            })
    return rows


def apply_spec(case, prediction, spec):
    if bool(spec.get("identity", False)):
        return case["labels"].copy(), case["instances"].copy(), 0
    logits = np.asarray(prediction["logits"], dtype=np.float64)
    benefit_logit = np.asarray(prediction["benefit_logit"], dtype=np.float64)
    benefit_probability = np.asarray(
        prediction["benefit_probability"], dtype=np.float64
    )
    move_probability = np.asarray(
        prediction["move_probability"], dtype=np.float64
    )
    candidates = np.asarray(case["target_instances"], dtype=np.int64) >= 0
    current = np.asarray(case["current_classes"], dtype=np.int64)
    support = np.asarray(case["neighbor_support"], dtype=np.float64)
    row = np.arange(len(current), dtype=np.int64)

    choice = (
        logits + float(spec["benefit_weight"]) * benefit_logit
        + 0.10 * support
    )
    choice[~candidates] = -1.0e9
    choice[row, current] = -1.0e9
    target_class = choice.argmax(axis=1)
    target_instance = np.asarray(case["target_instances"], dtype=np.int64)[
        row, target_class
    ]
    class_margin = logits[row, target_class] - logits[row, current]
    target_benefit = benefit_probability[row, target_class]
    eligible = (
        (target_instance >= 0)
        & (move_probability >= float(spec["move_minimum"]))
        & (target_benefit >= float(spec["benefit_minimum"]))
        & (class_margin >= float(spec["margin_minimum"]))
        & ~np.asarray(case["protected_local"], dtype=bool)
    )
    eligible = query_graph.component_filter(
        eligible, current, target_class, np.asarray(case["local_edges"]),
        int(spec["minimum_component"]),
    )
    priority = (
        np.log(np.clip(move_probability, 1.0e-8, 1.0))
        + np.log(np.clip(target_benefit, 1.0e-8, 1.0))
        + 0.25 * class_margin + 0.10 * support[row, target_class]
    )
    chosen = np.flatnonzero(eligible)
    cap = max(1, int(round(float(spec["cap_fraction"]) * len(case["instances"]))))
    if len(chosen) > cap:
        order = np.argsort(priority[chosen], kind="mergesort")
        chosen = chosen[order[-cap:]]
    output_instances = np.asarray(case["instances"], dtype=np.int64).copy()
    output_instances[np.asarray(case["band_indices"], dtype=np.int64)[chosen]] = (
        target_instance[chosen]
    )
    return query_graph.remap_labels(case, output_instances), output_instances, int(len(chosen))


def assignment_stats(cases, predictions):
    result = BASE_ASSIGNMENT_STATS(cases, predictions)
    true_positive = false_positive = false_negative = true_negative = 0
    for case, prediction in zip(cases, predictions):
        current = np.asarray(case["current_classes"], dtype=np.int64)
        target = np.asarray(case["target_classes"], dtype=np.int64)
        valid = np.asarray(case["assignment_valid"], dtype=bool)
        truth = (current != target) & valid
        guess = np.asarray(prediction["move_probability"]) >= 0.5
        true_positive += int(np.sum(guess & truth))
        false_positive += int(np.sum(guess & ~truth & valid))
        false_negative += int(np.sum(~guess & truth))
        true_negative += int(np.sum(~guess & ~truth & valid))
    result.update({
        "move_precision_at_0_5": float(true_positive / max(true_positive + false_positive, 1)),
        "move_recall_at_0_5": float(true_positive / max(true_positive + false_negative, 1)),
        "move_true_positive": true_positive,
        "move_false_positive": false_positive,
        "move_false_negative": false_negative,
        "move_true_negative": true_negative,
    })
    return result


def main():
    if os.environ.get("TRAIN_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    query_graph.SCHEMA = SCHEMA
    query_graph.BoundaryOwnershipQueryGraphNet = BoundaryOwnershipAssignmentGateNet
    query_graph.tensors = tensors
    query_graph.calculate_loss = calculate_loss
    query_graph.predict = predict
    query_graph.validation_specs = validation_specs
    query_graph.apply_spec = apply_spec
    query_graph.assignment_stats = assignment_stats
    query_graph.main()


if __name__ == "__main__":
    main()


