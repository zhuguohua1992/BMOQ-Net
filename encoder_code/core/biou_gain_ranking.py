# 本文件用于实现完整模型的边界编码、训练或推理。
import numpy as np
import torch
import torch.nn.functional as F

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import gcbl_bfanet as gcbl_bfanet
import multiview_guided_gcbl as multiview_guided


SCHEMA = "boundary-exact-one-ring-biou-gain-ranking"
                                                                               
                                                                               
                                                                     
GAIN_SCALE = 1000.0
GAIN_CLIP = 0.95
_BASE_APPLY_SPEC = edge_gated.apply_spec


def _directed_edges(edges):
    return (
        np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64),
        np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64),
    )


def _instance_labels(labels, instances):
    result = {}
    for instance_id in np.unique(instances):
        values, counts = np.unique(
            labels[instances == int(instance_id)], return_counts=True
        )
        result[int(instance_id)] = int(values[int(np.argmax(counts))])
    return result


def exact_biou_gain_targets(case):
                                                                                
    labels = np.asarray(case["labels"], dtype=np.int64)
    ground_truth = np.asarray(case["gt_labels"], dtype=np.int64)
    instances = np.asarray(case["instances"], dtype=np.int64)
    target_instances = np.asarray(case["target_instances"], dtype=np.int64)
    band_indices = np.asarray(case["band_indices"], dtype=np.int64)
    edges = np.asarray(case["edges"], dtype=np.int64)

    instance_label = _instance_labels(labels, instances)
    candidate_label = np.full(target_instances.shape, -1, dtype=np.int64)
    for instance_id in np.unique(target_instances[target_instances >= 0]):
        candidate_label[target_instances == int(instance_id)] = instance_label[
            int(instance_id)
        ]

    source, target = _directed_edges(edges)
    old_edge_boundary = labels[source] != labels[target]
    old_boundary_count = np.bincount(
        source,
        weights=old_edge_boundary.astype(np.float32),
        minlength=len(labels),
    )
    old_boundary = old_boundary_count > 0.0
    gt_edge_boundary = ground_truth[source] != ground_truth[target]
    gt_boundary = np.bincount(
        source,
        weights=gt_edge_boundary.astype(np.float32),
        minlength=len(labels),
    ) > 0.0
    current_intersection = float(np.logical_and(old_boundary, gt_boundary).sum())
    current_union = float(np.logical_or(old_boundary, gt_boundary).sum())
    current_biou = current_intersection / max(current_union, 1.0e-8)

    global_to_local = np.full(len(labels), -1, dtype=np.int64)
    global_to_local[band_indices] = np.arange(len(band_indices), dtype=np.int64)
    local_source = global_to_local[source]
    keep = local_source >= 0
    local_source = local_source[keep]
    global_source = source[keep]
    global_target = target[keep]
    old_edge_boundary = old_edge_boundary[keep]

    gain = np.zeros(target_instances.shape, dtype=np.float32)
    raw_positive = 0
    raw_negative = 0
    for class_id in range(target_instances.shape[1]):
        proposed_all = candidate_label[:, class_id]
        valid_row = proposed_all >= 0
        no_change = proposed_all == labels[band_indices]
        active_row = valid_row & ~no_change
        if not np.any(active_row):
            continue

        proposed_edge = proposed_all[local_source]
        active_edge = active_row[local_source]

        self_after_count = np.bincount(
            local_source[active_edge],
            weights=(
                proposed_edge[active_edge] != labels[global_target[active_edge]]
            ).astype(np.float32),
            minlength=len(band_indices),
        )
        self_after = self_after_count > 0.0
        self_delta = (
            self_after.astype(np.float32)
            - old_boundary[band_indices].astype(np.float32)
        )
        self_delta[~active_row] = 0.0

        neighbor_after_count = (
            old_boundary_count[global_target[active_edge]]
            - old_edge_boundary[active_edge].astype(np.float32)
            + (
                labels[global_target[active_edge]] != proposed_edge[active_edge]
            ).astype(np.float32)
        )
        neighbor_after = neighbor_after_count > 0.0
        neighbor_delta_edge = (
            neighbor_after.astype(np.float32)
            - old_boundary[global_target[active_edge]].astype(np.float32)
        )
        neighbor_delta = np.bincount(
            local_source[active_edge],
            weights=neighbor_delta_edge,
            minlength=len(band_indices),
        ).astype(np.float32)
        neighbor_intersection_delta = np.bincount(
            local_source[active_edge],
            weights=(
                neighbor_delta_edge
                * gt_boundary[global_target[active_edge]].astype(np.float32)
            ),
            minlength=len(band_indices),
        ).astype(np.float32)

        predicted_count_delta = self_delta + neighbor_delta
        intersection_delta = (
            self_delta * gt_boundary[band_indices].astype(np.float32)
            + neighbor_intersection_delta
        )
        after_union = (
            current_union + predicted_count_delta - intersection_delta
        )
        raw = (
            (current_intersection + intersection_delta)
            / np.maximum(after_union, 1.0e-8)
            - current_biou
        ).astype(np.float32)
        raw[~active_row] = 0.0
        gain[:, class_id] = np.clip(
            raw * GAIN_SCALE, -GAIN_CLIP, GAIN_CLIP
        ).astype(np.float32)
        raw_positive += int(np.sum(raw[active_row] > 0.0))
        raw_negative += int(np.sum(raw[active_row] < 0.0))

    valid = target_instances >= 0
    valid[np.arange(len(valid)), case["current_classes"]] = False
    case["gain_target"] = gain
    case["gain_valid"] = valid
    case["exact_gain_positive_count"] = raw_positive
    case["exact_gain_negative_count"] = raw_negative
    return case


def _exact_ranking_losses(gain, values):
    valid = values["gain_valid"].bool()
    target = values["gain_target"].float()
    if not valid.any():
        zero = gain.sum() * 0.0
        return zero, zero, zero

                                                                            
                                                              
    row_has_choices = valid.sum(dim=1) >= 2
    if row_has_choices.any():
        masked_target = (target / 0.15).masked_fill(~valid, -1.0e4)
        target_distribution = torch.softmax(masked_target, dim=1).detach()
        prediction_log_probability = torch.log_softmax(
            gain.masked_fill(~valid, -1.0e4), dim=1
        )
        listwise_each = -(
            target_distribution * prediction_log_probability
        ).sum(dim=1)
        listwise = listwise_each[row_has_choices].mean()
    else:
        listwise = gain.sum() * 0.0

                                                                         
                                                                             
    beneficial = (target > 0.0) & valid
    sign_each = F.binary_cross_entropy_with_logits(
        gain[valid], beneficial[valid].float(), reduction="none"
    )
    sign = boundary_tversky.balanced_mean(sign_each, beneficial[valid])

                                                                          
                                                                        
    non_positive = valid & ~beneficial
    positive_index = torch.nonzero(beneficial.reshape(-1), as_tuple=False).flatten()
    negative_index = torch.nonzero(non_positive.reshape(-1), as_tuple=False).flatten()
    if positive_index.numel() and negative_index.numel():
        flat_gain = gain.reshape(-1)
        flat_target = target.reshape(-1)
        pair_count = int(min(512, positive_index.numel(), negative_index.numel()))
        positive_top = positive_index[
            torch.topk(flat_target[positive_index], pair_count).indices
        ]
        negative_top = negative_index[
            torch.topk(flat_gain[negative_index].detach(), pair_count).indices
        ]
        pairwise = F.softplus(
            0.20 - (flat_gain[positive_top] - flat_gain[negative_top])
        ).mean()
    else:
        pairwise = gain.sum() * 0.0
    return listwise, sign, pairwise


def calculate_loss(logits, boundary_logit, gain, edge_logit, values):
    base, row = gcbl_bfanet.calculate_loss(
        logits, boundary_logit, gain, edge_logit, values
    )
    listwise, sign, pairwise = _exact_ranking_losses(gain, values)
    total = base + 0.25 * listwise + 0.30 * sign + 0.40 * pairwise
    row.update({
        "total": float(total.detach().cpu()),
        "base": float(base.detach().cpu()),
        "exact_gain_listwise": float(listwise.detach().cpu()),
        "exact_gain_balanced_sign": float(sign.detach().cpu()),
        "exact_gain_hard_pairwise": float(pairwise.detach().cpu()),
        "exact_gain_positive_fraction": float(
            ((values["gain_target"] > 0.0) & values["gain_valid"]).float().mean().detach().cpu()
        ),
    })
    return total, row


def apply_spec(case, prediction, spec):
                                                                            
    if not bool(spec.get("individual_actions", False)):
        return _BASE_APPLY_SPEC(case, prediction, spec)

    probability = prediction["probability"]
    gain = prediction["gain"]
    edge_gain = prediction["edge_gain"]
    current_class = case["current_classes"]
    target_instances = case["target_instances"]
    support_matrix = case["neighbor_support"]
    valid = target_instances >= 0
    valid[np.arange(len(valid)), current_class] = False
    if bool(spec.get("rank_by_gain", False)):
        choice = gain.copy()
    else:
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
        predicted_gain
        + 0.02 * target_probability
        + 0.005 * support
        + 0.002 * margin
    )
    chosen = np.flatnonzero(eligible)
    cap = max(1, int(round(float(spec["cap_fraction"]) * len(case["instances"]))))
    if len(chosen) > cap:
        chosen = chosen[
            np.argsort(score[chosen], kind="mergesort")[-cap:]
        ]

    output = case["instances"].copy()
    for local_index in chosen:
        output[int(case["band_indices"][local_index])] = int(
            target_instance[local_index]
        )
    mapping = {}
    for instance_id in np.unique(case["instances"]):
        mask = case["instances"] == int(instance_id)
        values, counts = np.unique(case["labels"][mask], return_counts=True)
        mapping[int(instance_id)] = int(values[int(np.argmax(counts))])
    maximum_instance_id = max(int(output.max()), max(mapping, default=0))
    lookup = np.zeros(maximum_instance_id + 1, dtype=np.int64)
    for instance_id, label in mapping.items():
        lookup[instance_id] = label
    return lookup[output], output, int(len(chosen))


def validation_specs():
                                                                            
    specs = list(boundary_tversky.validation_specs())
    seen = {tuple(sorted(spec.items())) for spec in specs}
    for probability_minimum in (0.0, 0.05):
        for cap_fraction in (
            0.0005, 0.00065, 0.00075, 0.00085, 0.001,
            0.00125, 0.0015, 0.002, 0.0025, 0.003,
            0.005, 0.01, 0.02,
        ):
            for gain_minimum in (-0.10, 0.0, 0.10, 0.25):
                for gain_weight in (2.0, 4.0, 8.0):
                    spec = {
                        "probability_minimum": probability_minimum,
                        "margin_minimum": -1.0,
                        "support_minimum": 0.0,
                        "cap_fraction": cap_fraction,
                        "minimum_component_size": 1,
                        "gain_weight": gain_weight,
                        "gain_minimum": gain_minimum,
                        "edge_gain_weight": 0.0,
                        "edge_gain_minimum": -1.0,
                        "individual_actions": True,
                        "rank_by_gain": True,
                    }
                    signature = tuple(sorted(spec.items()))
                    if signature not in seen:
                        seen.add(signature)
                        specs.append(spec)
    return specs


def _metric_cache(case):
    labels = np.asarray(case["labels"], dtype=np.int64)
    ground_truth = np.asarray(case["gt_labels"], dtype=np.int64)
    edges = np.asarray(case["edges"], dtype=np.int64)
    source, target = _directed_edges(edges)
    order = np.argsort(source, kind="mergesort")
    source = source[order]
    target = target[order]
    offsets = np.searchsorted(
        source, np.arange(len(labels) + 1, dtype=np.int64)
    )
    predicted_boundary = edge_gated.mesh_multiclass.boundary_vertices(labels, edges)
    gt_boundary = edge_gated.mesh_multiclass.boundary_vertices(ground_truth, edges)
    intersection = int(np.logical_and(predicted_boundary, gt_boundary).sum())
    union = int(np.logical_or(predicted_boundary, gt_boundary).sum())

    prediction_values = np.unique(labels)
    gt_values = np.unique(ground_truth)
    prediction_index = np.searchsorted(prediction_values, labels)
    gt_index = np.searchsorted(gt_values, ground_truth)
    contingency = np.bincount(
        prediction_index * len(gt_values) + gt_index,
        minlength=len(prediction_values) * len(gt_values),
    ).reshape(len(prediction_values), len(gt_values)).astype(np.int64)
    gt_counts = np.bincount(gt_index, minlength=len(gt_values)).astype(np.int64)
    return {
        "labels": labels,
        "ground_truth": ground_truth,
        "neighbors": target,
        "offsets": offsets,
        "predicted_boundary": predicted_boundary,
        "gt_boundary": gt_boundary,
        "intersection": intersection,
        "union": union,
        "prediction_values": prediction_values,
        "gt_values": gt_values,
        "contingency": contingency,
        "gt_counts": gt_counts,
    }


def _incremental_metrics(output, cache):
    labels = cache["labels"]
    changed = np.flatnonzero(output != labels)
    if not len(changed):
        boundary = cache["intersection"] / max(cache["union"], 1)
        return float(boundary), float(edge_gated.mesh_multiclass.instance_iou(
            cache["ground_truth"], labels
        ))

    neighbor_parts = [
        cache["neighbors"][cache["offsets"][index]:cache["offsets"][index + 1]]
        for index in changed
    ]
    affected = np.unique(np.concatenate([changed] + neighbor_parts))
    new_boundary = np.zeros(len(affected), dtype=bool)
    for position, vertex in enumerate(affected):
        neighbors = cache["neighbors"][
            cache["offsets"][vertex]:cache["offsets"][vertex + 1]
        ]
        if len(neighbors):
            new_boundary[position] = np.any(output[vertex] != output[neighbors])
    old_boundary = cache["predicted_boundary"][affected]
    gt_boundary = cache["gt_boundary"][affected]
    intersection = (
        cache["intersection"]
        - int(np.logical_and(old_boundary, gt_boundary).sum())
        + int(np.logical_and(new_boundary, gt_boundary).sum())
    )
    union = (
        cache["union"]
        - int(np.logical_or(old_boundary, gt_boundary).sum())
        + int(np.logical_or(new_boundary, gt_boundary).sum())
    )
    boundary_iou = intersection / max(union, 1)

    prediction_values = cache["prediction_values"]
    old_index = np.searchsorted(prediction_values, labels[changed])
    new_index = np.searchsorted(prediction_values, output[changed])
    if (
        np.any(new_index >= len(prediction_values))
        or np.any(prediction_values[new_index] != output[changed])
    ):
        return float(boundary_iou), float(edge_gated.mesh_multiclass.instance_iou(
            cache["ground_truth"], output
        ))
    gt_index = np.searchsorted(
        cache["gt_values"], cache["ground_truth"][changed]
    )
    contingency = cache["contingency"].copy()
    np.add.at(contingency, (old_index, gt_index), -1)
    np.add.at(contingency, (new_index, gt_index), 1)
    predicted_counts = contingency.sum(axis=1)
    active = (prediction_values != 0) & (predicted_counts > 0)
    if not np.any(active):
        instance_iou = 0.0
    else:
        active_rows = contingency[active]
        matched = np.argmax(active_rows, axis=1)
        intersection_counts = active_rows[
            np.arange(len(active_rows)), matched
        ]
        unions = (
            predicted_counts[active]
            + cache["gt_counts"][matched]
            - intersection_counts
        )
        instance_iou = float(np.mean(
            intersection_counts / np.maximum(unions, 1)
        ))
    return float(boundary_iou), float(instance_iou)


def select_validation(cases, predictions):
    caches = [_metric_cache(case) for case in cases]
    baseline_biou = []
    baseline_iou = []
    for case, cache in zip(cases, caches):
        biou, iou = _incremental_metrics(case["labels"], cache)
        baseline_biou.append(biou)
        baseline_iou.append(iou)
    baseline = {
        "BIoU": float(np.mean(baseline_biou)),
        "IoU": float(np.mean(baseline_iou)),
    }
    table = []
    for spec in validation_specs():
        boundary_rows = []
        iou_rows = []
        changed = []
        for case, prediction, cache in zip(cases, predictions, caches):
            labels, _instances, count = apply_spec(case, prediction, spec)
            biou, iou = _incremental_metrics(labels, cache)
            boundary_rows.append(biou)
            iou_rows.append(iou)
            changed.append(count)
        metrics = {
            "BIoU": float(np.mean(boundary_rows)),
            "IoU": float(np.mean(iou_rows)),
        }
        table.append({
            **spec,
            **metrics,
            "total_changed": int(sum(changed)),
            "passes_iou_guard": metrics["IoU"] >= baseline["IoU"] - 0.001,
        })
    eligible = [row for row in table if row["passes_iou_guard"]]
    selected = max(
        eligible,
        key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed"]),
    )
    return baseline, selected, table


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.mesh_multiclass.build_case = multiview_guided.build_case
    edge_gated.add_gain_targets = exact_biou_gain_targets
    edge_gated.EdgeGatedGainAwareMeshGNN = multiview_guided.MultiViewGuidedBFANetGNN
    edge_gated.calculate_loss = calculate_loss
    edge_gated.apply_spec = apply_spec
    edge_gated.select_validation = select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

