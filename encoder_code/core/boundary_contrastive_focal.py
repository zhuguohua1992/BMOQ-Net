# 本文件用于实现完整模型的边界编码、训练或推理。
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import multiscale_boundary_semantic as multiscale_boundary


SCHEMA = "boundary-boundary-contrastive-focal-multiscale"


def neighbour_softnn(features, labels, source, target, temperature=0.20):
                                                                                  
    if source.numel() == 0:
        return features.sum() * 0.0
    normalized = F.normalize(features.float(), dim=1)
    similarity = (normalized[source] * normalized[target]).sum(dim=1)
    affinity = torch.exp(similarity / float(temperature))
    positive = labels[source] == labels[target]

    count = features.shape[0]
    denominator = affinity.new_zeros(count)
    numerator = affinity.new_zeros(count)
    positive_count = affinity.new_zeros(count)
    negative_count = affinity.new_zeros(count)
    denominator.index_add_(0, source, affinity)
    numerator.index_add_(0, source, affinity * positive.float())
    positive_count.index_add_(0, source, positive.float())
    negative_count.index_add_(0, source, (~positive).float())
    eligible = (positive_count > 0.0) & (negative_count > 0.0)
    if not eligible.any():
        return features.sum() * 0.0
    probability = (numerator[eligible] + 1.0e-8) / (
        denominator[eligible] + 1.0e-8
    )
    return -torch.log(probability.clamp(min=1.0e-8)).mean()


def neighbourhood_average(features, source, target):
    if source.numel() == 0:
        return features
    aggregate = torch.zeros_like(features)
    degree = features.new_zeros(features.shape[0])
    aggregate.index_add_(0, source, features[target])
    degree.index_add_(0, source, torch.ones_like(source, dtype=features.dtype))
    aggregate = aggregate / degree.clamp(min=1.0).unsqueeze(1)
    return 0.5 * features + 0.5 * aggregate


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = boundary_tversky.calculate_loss(logits, boundary_logit, gain, edge_logit, values)
    target_class = values["target_class"]
    source = values["source"].long()
    target = values["target"].long()

                                                                           
                                                                              
                                                                           
    contrast_one = neighbour_softnn(logits, target_class, source, target)
    contextual_logits = neighbourhood_average(logits, source, target)
    contrast_context = neighbour_softnn(
        contextual_logits, target_class, source, target
    )
    contrastive = 0.5 * (contrast_one + contrast_context)

                                                                       
                                                                         
                                             
    probability = torch.softmax(logits, dim=1)
    target_probability = probability.gather(1, target_class[:, None]).squeeze(1)
    boundary_mask = values["gt_boundary"] > 0.5
    if boundary_mask.any():
        boundary_focal = (
            torch.pow(1.0 - target_probability[boundary_mask], 2.0)
            * -torch.log(target_probability[boundary_mask].clamp(min=1.0e-8))
        ).mean()
    else:
        boundary_focal = logits.sum() * 0.0

    total = base + 0.08 * contrastive + 0.06 * boundary_focal
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "boundary_contrastive": float(contrastive.detach().cpu()),
        "boundary_focal": float(boundary_focal.detach().cpu()),
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

