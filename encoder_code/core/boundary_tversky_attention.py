# 本文件用于实现完整模型的边界编码、训练或推理。
import numpy as np
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import mesh_attention_gnn as mesh_attention


SCHEMA = "boundary-boundary-tversky-mesh-attention"


def soft_tversky(probability, target, alpha=0.35, beta=0.65, epsilon=1.0e-6):
    probability = probability.float()
    target = target.float()
    true_positive = (probability * target).sum()
    false_positive = (probability * (1.0 - target)).sum()
    false_negative = ((1.0 - probability) * target).sum()
    score = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon
    )
    return 1.0 - score


def balanced_mean(values, changed):
    changed = changed.bool()
    positive = values[changed]
    negative = values[~changed]
    terms = []
    if positive.numel():
        terms.append(positive.mean())
    if negative.numel():
        terms.append(negative.mean())
    return sum(terms) / max(len(terms), 1)


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    target = values["target_class"]
    current = values["current"]
    changed = target != current
    probability = torch.softmax(logits, dim=1)
    log_probability = F.log_softmax(logits, dim=1)

    target_probability = probability.gather(1, target[:, None]).squeeze(1)
    current_probability = probability.gather(1, current[:, None]).squeeze(1)
    nll = -log_probability.gather(1, target[:, None]).squeeze(1)
    smoothing = -log_probability.mean(dim=1)
    focal = torch.pow(1.0 - target_probability, 1.25)
    semantic = balanced_mean((0.98 * nll + 0.02 * smoothing) * focal, changed)

                                                                           
                                                                             
    change_probability = (1.0 - current_probability).clamp(1.0e-5, 1.0 - 1.0e-5)
    change_target = changed.float()
    action_bce = balanced_mean(
        F.binary_cross_entropy(change_probability, change_target, reduction="none"),
        changed,
    )
    action_tversky = soft_tversky(change_probability, change_target)

    other_probability = probability.masked_fill(
        F.one_hot(current, num_classes=probability.shape[1]).bool(), -1.0
    ).max(dim=1).values
    changed_margin = F.relu(0.20 - (target_probability - current_probability))
    retained_margin = F.relu(0.10 - (current_probability - other_probability))
    margin_loss = balanced_mean(torch.where(changed, changed_margin, retained_margin), changed)

    gt_boundary = values["gt_boundary"]
    boundary_probability = torch.sigmoid(boundary_logit)
    boundary_bce = balanced_mean(
        F.binary_cross_entropy_with_logits(boundary_logit, gt_boundary, reduction="none"),
        gt_boundary > 0.5,
    )
    boundary_tversky = soft_tversky(boundary_probability, gt_boundary, alpha=0.45, beta=0.55)

    edge_u, edge_v = values["edge_u"], values["edge_v"]
    edge_target = (target[edge_u] != target[edge_v]).float()
    edge_probability = torch.sigmoid(edge_logit)
    edge_bce = balanced_mean(
        F.binary_cross_entropy_with_logits(edge_logit, edge_target, reduction="none"),
        edge_target > 0.5,
    )
    edge_transition = (
        edge_bce
        + edge_gated.mesh_multiclass.soft_iou_loss(edge_probability, edge_target)
        + soft_tversky(edge_probability, edge_target, alpha=0.45, beta=0.55)
    )

    valid = values["gain_valid"]
    if valid.any():
        gain_regression = F.smooth_l1_loss(gain[valid], values["gain_target"][valid])
        positive_gain = (values["gain_target"][valid] > 0.0).float()
        gain_sign = F.binary_cross_entropy_with_logits(4.0 * gain[valid], positive_gain)
    else:
        gain_regression = gain.sum() * 0.0
        gain_sign = gain.sum() * 0.0

    total = (
        semantic
        + 0.55 * action_bce
        + 0.75 * action_tversky
        + 0.35 * margin_loss
        + 0.30 * boundary_bce
        + 0.35 * boundary_tversky
        + 1.35 * edge_transition
        + 0.60 * gain_regression
        + 0.15 * gain_sign
    )
    return total, {
        "total": float(total.detach().cpu()),
        "semantic": float(semantic.detach().cpu()),
        "action_bce": float(action_bce.detach().cpu()),
        "action_tversky": float(action_tversky.detach().cpu()),
        "margin": float(margin_loss.detach().cpu()),
        "boundary": float((boundary_bce + boundary_tversky).detach().cpu()),
        "gain_regression": float(gain_regression.detach().cpu()),
        "gain_sign": float(gain_sign.detach().cpu()),
        "edge_transition": float(edge_transition.detach().cpu()),
        "change_fraction": float(changed.float().mean().detach().cpu()),
    }


def validation_specs():
    rng = np.random.RandomState(2026139)
    choices = {
        "probability_minimum": (0.10, 0.20, 0.35, 0.50, 0.70),
        "margin_minimum": (-0.30, -0.10, 0.0, 0.10),
        "support_minimum": (0.0, 0.05, 0.10, 0.20),
        "cap_fraction": (0.002, 0.01, 0.05, 0.10, 0.20),
        "minimum_component_size": (1, 2, 3, 5),
        "gain_weight": (0.50, 1.0, 2.0),
        "gain_minimum": (-0.30, -0.10, 0.0, 0.10),
        "edge_gain_weight": (0.50, 1.0, 2.0),
        "edge_gain_minimum": (-0.30, -0.10, 0.0, 0.10),
    }
    specs = [{
        "probability_minimum": 0.20,
        "margin_minimum": 0.0,
        "support_minimum": 0.0,
        "cap_fraction": 0.20,
        "minimum_component_size": 3,
        "gain_weight": 2.0,
        "gain_minimum": 0.0,
        "edge_gain_weight": 0.50,
        "edge_gain_minimum": -0.30,
    }]
    seen = {tuple(sorted(specs[0].items()))}
    while len(specs) < 320:
        spec = {key: values[int(rng.randint(len(values)))] for key, values in choices.items()}
        signature = tuple(sorted(spec.items()))
        if signature not in seen:
            seen.add(signature)
            specs.append(spec)
    return specs


def select_validation(cases, predictions):
    baseline = {
        "BIoU": float(np.mean([
            edge_gated.mesh_multiclass.boundary_iou(case["gt_labels"], case["labels"], case["edges"])
            for case in cases
        ])),
        "IoU": float(np.mean([
            edge_gated.mesh_multiclass.instance_iou(case["gt_labels"], case["labels"])
            for case in cases
        ])),
    }
    table = []
    for spec in validation_specs():
        metrics, _, changed = edge_gated.evaluate(cases, predictions, spec)
        table.append({
            **spec,
            **metrics,
            "total_changed": int(sum(changed)),
            "passes_iou_guard": metrics["IoU"] >= baseline["IoU"] - 0.001,
        })
    eligible = [row for row in table if row["passes_iou_guard"]]
    selected = max(eligible, key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed"]))
    return baseline, selected, table


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.EdgeGatedGainAwareMeshGNN = mesh_attention.MeshAttentionGainGNN
    edge_gated.calculate_loss = calculate_loss
    edge_gated.select_validation = select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

