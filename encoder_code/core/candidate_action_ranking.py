# 本文件用于实现完整模型的边界编码、训练或推理。
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import multiscale_boundary_semantic as multiscale_boundary
import boundary_displacement_focal as boundary_displacement


SCHEMA = "boundary-candidate-action-ranking-multiscale"


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = boundary_displacement.calculate_loss(logits, boundary_logit, gain, edge_logit, values)
    target = values["target_class"].long()
    current = values["current"].long()
    changed = target != current
    valid = values["gain_valid"].bool()
    vertex = torch.arange(len(target), device=logits.device)

                                                                           
                                                                           
    action_score = logits + 0.50 * gain
    action_score = action_score.clone()
    action_score[vertex, current] = logits[vertex, current]
    candidate = valid.clone()
    candidate[vertex, current] = True

    reachable = candidate[vertex, target]
    if reachable.any():
        masked_score = action_score.masked_fill(~candidate, -1.0e4)
        listwise_each = F.cross_entropy(
            masked_score[reachable], target[reachable], reduction="none"
        )
        listwise = boundary_tversky.balanced_mean(listwise_each, changed[reachable])
    else:
        listwise = logits.sum() * 0.0

                                                                             
                                                                             
                                                                     
    target_move = F.one_hot(target, num_classes=logits.shape[1]).bool() & valid
    if valid.any():
        current_score = logits[vertex, current].unsqueeze(1)
        move_logit = action_score - current_score
        pair_each = F.binary_cross_entropy_with_logits(
            move_logit[valid], target_move[valid].float(), reduction="none"
        )
        pairwise = boundary_tversky.balanced_mean(pair_each, target_move[valid])
    else:
        pairwise = logits.sum() * 0.0

    total = base + 0.35 * listwise + 0.45 * pairwise
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "candidate_listwise": float(listwise.detach().cpu()),
        "candidate_pairwise": float(pairwise.detach().cpu()),
        "candidate_reachable_fraction": float(reachable.float().mean().detach().cpu()),
        "candidate_positive_fraction": float(target_move.float().mean().detach().cpu()),
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

