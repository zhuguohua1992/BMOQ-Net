# 本文件用于实现完整模型的边界编码、训练或推理。
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import soft_semantic_boundary_iou as boundary_soft_semantic
import bfanet_cross_attention as bfanet_cross_attention


SCHEMA = "boundary-gcbl-bfanet-guided-boundary"
_ACTIVE_MODEL = None


class GuidedBoundaryBFANetGNN(bfanet_cross_attention.BFANetMeshCrossAttentionGNN):
                                                                         

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__(input_dim, hidden_dim, layers, dropout)
        global _ACTIVE_MODEL
        _ACTIVE_MODEL = self
        self.cached_semantic_hidden = None
        self.class_head.register_forward_pre_hook(self._cache_classifier_input)

    def _cache_classifier_input(self, _module, inputs):
        self.cached_semantic_hidden = inputs[0]


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = boundary_soft_semantic.calculate_loss(
        logits, boundary_logit, gain, edge_logit, values
    )
    model = _ACTIVE_MODEL
    if model is None or model.cached_semantic_hidden is None:
        raise RuntimeError("B semantic embedding was not captured")

    hidden = F.normalize(model.cached_semantic_hidden.float(), dim=1)
    prototypes = F.normalize(model.class_head.weight.float(), dim=1)
    target_class = values["target_class"].long()
    source = values["source"].long()
    target = values["target"].long()
    boundary_mask = values["gt_boundary"] > 0.5

    probability = torch.softmax(logits.float(), dim=1)
    target_probability = probability.gather(
        1, target_class[:, None]
    ).squeeze(1)
    detached_confidence = target_probability.detach()

                                                                           
                                                                           
                                                             
    cross = target_class[source] != target_class[target]
    cross = cross & (boundary_mask[source] | boundary_mask[target])
    if cross.any():
        cross_source = source[cross]
        cross_target = target[cross]
        pair_confidence = torch.sqrt(
            detached_confidence[cross_source]
            * detached_confidence[cross_target]
        )
        source_uncertainty = 1.0 - detached_confidence[cross_source]
        pair_weight = (0.25 + 0.75 * pair_confidence) * (
            0.75 + 0.25 * source_uncertainty
        )

        feature_cosine = (
            hidden[cross_source] * hidden[cross_target]
        ).sum(dim=1)
        feature_separation = torch.relu(feature_cosine - 0.05)

        own_prototype = prototypes[target_class[cross_source]]
        other_prototype = prototypes[target_class[cross_target]]
        own_similarity = (hidden[cross_source] * own_prototype).sum(dim=1)
        other_similarity = (hidden[cross_source] * other_prototype).sum(dim=1)
        asymmetric_margin = torch.relu(
            other_similarity - own_similarity + 0.20
        )
        guided_inter = weighted_mean(
            0.5 * feature_separation + 0.5 * asymmetric_margin,
            pair_weight,
        )
    else:
        guided_inter = logits.sum() * 0.0

                                                                            
                                                                             
                           
    if boundary_mask.any():
        correct_prototype = prototypes[target_class[boundary_mask]]
        prototype_cosine = (
            hidden[boundary_mask] * correct_prototype
        ).sum(dim=1)
        prototype_weight = 0.5 + (
            1.0 - detached_confidence[boundary_mask]
        )
        guided_intra = weighted_mean(
            1.0 - prototype_cosine, prototype_weight
        )
        boundary_focal = weighted_mean(
            -torch.log(target_probability[boundary_mask].clamp_min(1.0e-8)),
            torch.pow(1.0 - target_probability[boundary_mask], 2.0),
        )
    else:
        guided_intra = logits.sum() * 0.0
        boundary_focal = logits.sum() * 0.0

    prototype_gram = prototypes @ prototypes.transpose(0, 1)
    off_diagonal = ~torch.eye(
        prototype_gram.shape[0],
        dtype=torch.bool,
        device=prototype_gram.device,
    )
    prototype_distribution = torch.relu(
        prototype_gram[off_diagonal] - 0.10
    ).pow(2).mean()

    total = (
        base
        + 0.06 * guided_inter
        + 0.04 * guided_intra
        + 0.05 * boundary_focal
        + 0.003 * prototype_distribution
    )
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "gcbl_guided_inter": float(guided_inter.detach().cpu()),
        "gcbl_guided_intra": float(guided_intra.detach().cpu()),
        "gcbl_boundary_focal": float(boundary_focal.detach().cpu()),
        "gcbl_prototype_distribution": float(
            prototype_distribution.detach().cpu()
        ),
        "gcbl_cross_edge_fraction": float(cross.float().mean().detach().cpu()),
    })
    return total, row


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.EdgeGatedGainAwareMeshGNN = GuidedBoundaryBFANetGNN
    edge_gated.calculate_loss = calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

