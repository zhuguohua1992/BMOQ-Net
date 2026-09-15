# 本文件用于实现完整模型的边界编码、训练或推理。
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import multiscale_boundary_semantic as multiscale_boundary
import candidate_action_ranking as candidate_action


SCHEMA = "boundary-soft-semantic-boundary-iou-multiscale"


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = candidate_action.calculate_loss(logits, boundary_logit, gain, edge_logit, values)
    probability = torch.softmax(logits, dim=1)
    edge_u = values["edge_u"].long()
    edge_v = values["edge_v"].long()
    target_class = values["target_class"].long()
    current = values["current"].long()

                                                                            
                                                                           
                                                                             
    same_probability = (probability[edge_u] * probability[edge_v]).sum(dim=1)
    transition_probability = (1.0 - same_probability).clamp(1.0e-5, 1.0 - 1.0e-5)
    transition_target = (target_class[edge_u] != target_class[edge_v]).float()

    transition_bce_each = F.binary_cross_entropy(
        transition_probability, transition_target, reduction="none"
    )
    transition_bce = boundary_tversky.balanced_mean(
        transition_bce_each, transition_target > 0.5
    )
    transition_tversky = boundary_tversky.soft_tversky(
        transition_probability, transition_target, alpha=0.45, beta=0.55
    )
    transition_iou = edge_gated.mesh_multiclass.soft_iou_loss(
        transition_probability, transition_target
    )

    current_transition = current[edge_u] != current[edge_v]
    displaced = current_transition != (transition_target > 0.5)
    true_probability = torch.where(
        transition_target > 0.5,
        transition_probability,
        1.0 - transition_probability,
    )
    focal_each = torch.pow(1.0 - true_probability, 2.0) * transition_bce_each
    if displaced.any():
        displaced_focal = focal_each[displaced].mean()
    else:
        displaced_focal = logits.sum() * 0.0

    total = (
        base
        + 0.45 * transition_bce
        + 0.55 * transition_tversky
        + 0.45 * transition_iou
        + 0.35 * displaced_focal
    )
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "semantic_transition_bce": float(transition_bce.detach().cpu()),
        "semantic_transition_tversky": float(transition_tversky.detach().cpu()),
        "semantic_transition_iou": float(transition_iou.detach().cpu()),
        "semantic_transition_displaced_focal": float(displaced_focal.detach().cpu()),
        "semantic_transition_displaced_fraction": float(displaced.float().mean().detach().cpu()),
    })
    return total, row


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.EdgeGatedGainAwareMeshGNN = multiscale_boundary.MultiScaleBoundarySemanticGNN
    edge_gated.calculate_loss = calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

