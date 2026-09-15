# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_boundary_label_refiner import (
    SCORE_KEYS,
    aggregate_metrics,
    boundary_vertices,
    build_features,
    expand_band,
    feature_names,
    parse_obj,
    scan_paths,
    sha256,
    unique_edges,
)


SCHEMA = "tgn-ambiguity-gated-boundary-graph-refiner"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def vertex_normals_and_curvature(vertices, faces, edges):
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    ).astype(np.float32)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(face_norms, 1e-8)
    normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    dot = np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1)
    edge_curvature = np.clip(1.0 - dot, 0.0, 2.0).astype(np.float32)
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    directed_curvature = np.concatenate((edge_curvature, edge_curvature))
    degree = np.bincount(source, minlength=len(vertices)).astype(np.float32)
    curvature_mean = np.bincount(
        source, weights=directed_curvature, minlength=len(vertices)
    ).astype(np.float32) / np.maximum(degree, 1.0)
    curvature_max = np.zeros(len(vertices), dtype=np.float32)
    np.maximum.at(curvature_max, source, directed_curvature)
    return normals, curvature_mean, curvature_max


def graph_distance(seed, edges, maximum_hops):
    distance = np.full(len(seed), maximum_hops + 1, dtype=np.int64)
    frontier = seed.copy()
    distance[seed] = 0
    visited = seed.copy()
    for hop in range(1, maximum_hops + 1):
        touches = frontier[edges[:, 0]] | frontier[edges[:, 1]]
        next_frontier = np.zeros(len(seed), dtype=bool)
        next_frontier[edges[touches].reshape(-1)] = True
        next_frontier &= ~visited
        if not next_frontier.any():
            break
        distance[next_frontier] = hop
        visited |= next_frontier
        frontier = next_frontier
    return distance


def list_prediction_paths(pred_roots, maximum_cases):
    paths = []
    for root in pred_roots:
        paths.extend(sorted(root.glob("*.json")))
    paths = sorted(paths)
    names = [path.name for path in paths]
    if not paths or len(names) != len(set(names)):
        raise RuntimeError("Prediction roots are empty or contain duplicate names")
    if maximum_cases > 0:
        paths = paths[:maximum_cases]
    return paths


def load_cases_limited(
    pred_roots,
    score_root,
    obj_root,
    json_root,
    band_hops,
    read_gt,
    maximum_cases,
    boundary_distance_hops,
):
    prediction_paths = list_prediction_paths(pred_roots, maximum_cases)
    cases = []
    for position, prediction_path in enumerate(prediction_paths):
        stem = prediction_path.stem
        obj_path, gt_path = scan_paths(stem, obj_root, json_root)
        score_path = score_root / (stem + ".npz")
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        labels = np.asarray(prediction["labels"], dtype=np.int64).reshape(-1)
        vertices, faces = parse_obj(obj_path)
        if len(vertices) != len(labels):
            raise ValueError("Vertex/prediction length mismatch {}".format(stem))
        built = build_features(vertices, faces, labels, score_path, band_hops)
        edges = built["edges"]
        band_indices = built["band_indices"]

        normals, curvature_mean, curvature_max = vertex_normals_and_curvature(
            vertices, faces, edges
        )
        coarse_boundary = boundary_vertices(labels, edges)
        coarse_distance = graph_distance(coarse_boundary, edges, max(band_hops, 1))
        primary = built["features"][:, list(SCORE_KEYS).index("frontal_smooth")]
        ambiguity = 1.0 - np.abs(np.clip(primary, 0.0, 1.0) * 2.0 - 1.0)
        geometry = np.column_stack(
            (
                normals[band_indices],
                np.clip(curvature_mean[band_indices] / 0.25, 0.0, 4.0),
                np.clip(curvature_max[band_indices] / 0.50, 0.0, 4.0),
                np.clip(coarse_distance[band_indices] / max(band_hops, 1), 0.0, 2.0),
                ambiguity,
            )
        ).astype(np.float32)

        global_to_local = np.full(len(labels), -1, dtype=np.int64)
        global_to_local[band_indices] = np.arange(len(band_indices), dtype=np.int64)
        inside = (global_to_local[edges[:, 0]] >= 0) & (
            global_to_local[edges[:, 1]] >= 0
        )
        local_edges = np.column_stack(
            (
                global_to_local[edges[inside, 0]],
                global_to_local[edges[inside, 1]],
            )
        ).astype(np.int64)
        if not len(local_edges):
            raise RuntimeError("No local band edges for {}".format(stem))
        local_min = int(local_edges.min())
        local_max = int(local_edges.max())
        if local_min < 0 or local_max >= len(band_indices):
            raise RuntimeError(
                "Local edge audit failed {} min={} max={} band={}".format(
                    stem, local_min, local_max, len(band_indices)
                )
            )

        ground_truth = None
        switch_target = None
        switch_valid = None
        gt_boundary_local = None
        gt_distance_local = None
        gt_edge_boundary = None
        if read_gt:
            ground_truth = np.asarray(
                json.loads(gt_path.read_text(encoding="utf-8"))["labels"],
                dtype=np.int64,
            ).reshape(-1)
            if len(ground_truth) != len(labels):
                raise ValueError("GT/prediction length mismatch {}".format(stem))
            gt_band = ground_truth[band_indices]
            current_band = labels[band_indices]
            alternative = built["alternative_labels"]
            switch_valid = (gt_band == current_band) | (gt_band == alternative)
            switch_target = (
                (gt_band == alternative) & (gt_band != current_band)
            ).astype(np.float32)
            gt_boundary = boundary_vertices(ground_truth, edges)
            gt_boundary_local = gt_boundary[band_indices].astype(np.float32)
            gt_distance = graph_distance(gt_boundary, edges, boundary_distance_hops)
            gt_distance_local = np.clip(
                gt_distance[band_indices] / max(boundary_distance_hops, 1), 0.0, 1.0
            ).astype(np.float32)
            gt_edge_boundary = (
                ground_truth[band_indices[local_edges[:, 0]]]
                != ground_truth[band_indices[local_edges[:, 1]]]
            ).astype(np.float32)

        cases.append(
            {
                "stem": stem,
                "prediction_path": prediction_path,
                "prediction": prediction,
                "labels": labels,
                "ground_truth": ground_truth,
                "score_path": score_path,
                "obj_path": obj_path,
                "features": built["features"],
                "geometry": geometry,
                "band_indices": band_indices,
                "alternative_labels": built["alternative_labels"],
                "edges": edges,
                "local_edges": local_edges,
                "switch_target": switch_target,
                "switch_valid": switch_valid,
                "gt_boundary_local": gt_boundary_local,
                "gt_distance_local": gt_distance_local,
                "gt_edge_boundary": gt_edge_boundary,
                "initial_boundary_count": built["initial_boundary_count"],
                "band_count": built["band_count"],
            }
        )
        print(
            "LOAD_GRAPH_CASE {}/{} {} vertices={} band={} local_edges={}".format(
                position + 1,
                len(prediction_paths),
                stem,
                len(labels),
                built["band_count"],
                len(local_edges),
            ),
            flush=True,
        )
    return cases


def fit_standardizer(cases):
    count = 0
    total = None
    squared = None
    for case in cases:
        features = case["features"].astype(np.float64)
        if total is None:
            total = features.sum(axis=0)
            squared = np.square(features).sum(axis=0)
        else:
            total += features.sum(axis=0)
            squared += np.square(features).sum(axis=0)
        count += len(features)
    mean = total / max(count, 1)
    variance = np.maximum(squared / max(count, 1) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    std[std < 1e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(cases, mean, std):
    for case in cases:
        case["features"] = np.clip(
            (case["features"] - mean[None, :]) / std[None, :], -8.0, 8.0
        ).astype(np.float32)


def make_directed_local_edges(local_edges):
    source = np.concatenate((local_edges[:, 0], local_edges[:, 1]))
    target = np.concatenate((local_edges[:, 1], local_edges[:, 0]))
    return source.astype(np.int64), target.astype(np.int64)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.message = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, source, target):
        neighbor = hidden[target]
        message = self.message(torch.cat((neighbor, neighbor - hidden[source]), dim=1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, source, message)
        degree = torch.zeros(
            len(hidden), dtype=hidden.dtype, device=hidden.device
        )
        degree.index_add_(0, source, torch.ones_like(source, dtype=hidden.dtype))
        aggregate = aggregate / degree.clamp(min=1.0).unsqueeze(1)
        update = self.update(torch.cat((hidden, aggregate), dim=1))
        return self.norm(hidden + self.dropout(update))


class AmbiguityGatedGraphRefiner(nn.Module):
    def __init__(
        self,
        boundary_dim,
        semantic_dim,
        hidden_dim,
        graph_layers,
        dropout,
    ):
        super().__init__()
        self.boundary_encoder = MLP(
            boundary_dim, hidden_dim, hidden_dim, dropout
        )
        self.semantic_encoder = MLP(
            semantic_dim, hidden_dim, hidden_dim, dropout
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.graph_blocks = nn.ModuleList(
            [GraphMessageBlock(hidden_dim, dropout) for _ in range(graph_layers)]
        )
        self.context = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.switch_head = nn.Linear(hidden_dim, 1)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.distance_head = nn.Linear(hidden_dim, 1)

    def forward(self, boundary_x, semantic_x, source, target):
        boundary_hidden = self.boundary_encoder(boundary_x)
        semantic_hidden = self.semantic_encoder(semantic_x)
        concatenated = torch.cat((boundary_hidden, semantic_hidden), dim=1)
        gate = self.gate(concatenated)
        hidden = gate * boundary_hidden + (1.0 - gate) * semantic_hidden
        hidden = self.fusion_norm(hidden + self.fusion(concatenated))
        for block in self.graph_blocks:
            hidden = block(hidden, source, target)
        mean_context = hidden.mean(dim=0, keepdim=True)
        max_context = hidden.max(dim=0, keepdim=True)[0]
        scan_context = self.context(torch.cat((mean_context, max_context), dim=1))
        hidden = self.output_norm(hidden + scan_context)
        return {
            "switch_logit": self.switch_head(hidden).squeeze(1),
            "boundary_logit": self.boundary_head(hidden).squeeze(1),
            "distance": torch.sigmoid(self.distance_head(hidden).squeeze(1)),
            "embedding": hidden,
            "gate_mean": gate.mean(),
        }


def case_to_tensors(case, device, boundary_feature_count):
    base = torch.from_numpy(case["features"]).to(device=device, dtype=torch.float32)
    geometry = torch.from_numpy(case["geometry"]).to(device=device, dtype=torch.float32)
    boundary_x = torch.cat((base[:, :boundary_feature_count], geometry), dim=1)
    semantic_x = base[:, boundary_feature_count:]
    source_np, target_np = make_directed_local_edges(case["local_edges"])
    result = {
        "boundary_x": boundary_x,
        "semantic_x": semantic_x,
        "source": torch.from_numpy(source_np).to(device=device, dtype=torch.long),
        "target": torch.from_numpy(target_np).to(device=device, dtype=torch.long),
        "edge_u": torch.from_numpy(case["local_edges"][:, 0]).to(
            device=device, dtype=torch.long
        ),
        "edge_v": torch.from_numpy(case["local_edges"][:, 1]).to(
            device=device, dtype=torch.long
        ),
        "current": torch.from_numpy(case["labels"][case["band_indices"]]).to(
            device=device, dtype=torch.long
        ),
        "alternative": torch.from_numpy(case["alternative_labels"]).to(
            device=device, dtype=torch.long
        ),
    }
    if case["ground_truth"] is not None:
        result.update(
            {
                "switch_target": torch.from_numpy(case["switch_target"]).to(
                    device=device, dtype=torch.float32
                ),
                "switch_valid": torch.from_numpy(case["switch_valid"]).to(
                    device=device, dtype=torch.bool
                ),
                "gt_boundary": torch.from_numpy(case["gt_boundary_local"]).to(
                    device=device, dtype=torch.float32
                ),
                "gt_distance": torch.from_numpy(case["gt_distance_local"]).to(
                    device=device, dtype=torch.float32
                ),
                "gt_edge_boundary": torch.from_numpy(case["gt_edge_boundary"]).to(
                    device=device, dtype=torch.float32
                ),
                "gt_band_labels": torch.from_numpy(
                    case["ground_truth"][case["band_indices"]]
                ).to(device=device, dtype=torch.long),
            }
        )
    return result


def weighted_bce_with_logits(logit, target, positive_weight):
    weight = torch.where(
        target > 0.5,
        torch.full_like(target, positive_weight),
        torch.ones_like(target),
    )
    return F.binary_cross_entropy_with_logits(logit, target, weight=weight)


def soft_iou_loss(probability, target):
    intersection = (probability * target).sum()
    union = probability.sum() + target.sum() - intersection
    return 1.0 - (intersection + 1.0) / (union + 1.0)


def edge_boundary_probability(probability, tensors):
    u = tensors["edge_u"]
    v = tensors["edge_v"]
    pu = probability[u]
    pv = probability[v]
    current_u = tensors["current"][u]
    current_v = tensors["current"][v]
    alternative_u = tensors["alternative"][u]
    alternative_v = tensors["alternative"][v]
    q00 = (1.0 - pu) * (1.0 - pv) * (current_u != current_v).float()
    q01 = (1.0 - pu) * pv * (current_u != alternative_v).float()
    q10 = pu * (1.0 - pv) * (alternative_u != current_v).float()
    q11 = pu * pv * (alternative_u != alternative_v).float()
    return (q00 + q01 + q10 + q11).clamp(1e-5, 1.0 - 1e-5)


def vertex_boundary_probability(edge_probability, tensors, vertex_count):
    log_no_boundary = torch.log1p(-edge_probability.clamp(max=1.0 - 1e-5))
    aggregate = torch.zeros(
        vertex_count, dtype=edge_probability.dtype, device=edge_probability.device
    )
    aggregate.index_add_(0, tensors["edge_u"], log_no_boundary)
    aggregate.index_add_(0, tensors["edge_v"], log_no_boundary)
    return (1.0 - torch.exp(aggregate)).clamp(1e-5, 1.0 - 1e-5)


def contrastive_edge_loss(embedding, tensors, maximum_edges, margin):
    u = tensors["edge_u"]
    v = tensors["edge_v"]
    vertex_count = len(embedding)
    label_count = len(tensors["gt_band_labels"])
    edge_min = int(torch.minimum(u.min(), v.min()).detach().cpu())
    edge_max = int(torch.maximum(u.max(), v.max()).detach().cpu())
    if edge_min < 0 or edge_max >= vertex_count or label_count != vertex_count:
        raise RuntimeError(
            "Contrastive index audit failed min={} max={} vertices={} labels={}".format(
                edge_min, edge_max, vertex_count, label_count
            )
        )
    if len(u) > maximum_edges:
                                                                              
                                                                             
                                                                          
        stride = int(math.ceil(len(u) / float(maximum_edges)))
        selected = torch.arange(0, len(u), stride, device=u.device)[:maximum_edges]
        u = u[selected]
        v = v[selected]
    normalized = F.normalize(embedding, dim=1)
    cosine = (normalized[u] * normalized[v]).sum(dim=1)
    same = tensors["gt_band_labels"][u] == tensors["gt_band_labels"][v]
    same_loss = (1.0 - cosine[same]).mean() if same.any() else cosine.sum() * 0.0
    different = ~same
    different_loss = (
        F.relu(cosine[different] - margin).mean()
        if different.any()
        else cosine.sum() * 0.0
    )
    return same_loss + different_loss


def calculate_loss(outputs, tensors, weights, positive_weights, contrastive_max_edges):
    valid = tensors["switch_valid"]
    switch_loss = weighted_bce_with_logits(
        outputs["switch_logit"][valid],
        tensors["switch_target"][valid],
        positive_weights["switch"],
    )
    auxiliary_boundary_loss = weighted_bce_with_logits(
        outputs["boundary_logit"],
        tensors["gt_boundary"],
        positive_weights["vertex_boundary"],
    ) + soft_iou_loss(torch.sigmoid(outputs["boundary_logit"]), tensors["gt_boundary"])
    distance_loss = F.smooth_l1_loss(outputs["distance"], tensors["gt_distance"])

    switch_probability = torch.sigmoid(outputs["switch_logit"])
    edge_probability = edge_boundary_probability(switch_probability, tensors)
    edge_target = tensors["gt_edge_boundary"]
    edge_weight = torch.where(
        edge_target > 0.5,
        torch.full_like(edge_target, positive_weights["edge_boundary"]),
        torch.ones_like(edge_target),
    )
    edge_bce = F.binary_cross_entropy(edge_probability, edge_target, weight=edge_weight)
    edge_iou = soft_iou_loss(edge_probability, edge_target)
    vertex_probability = vertex_boundary_probability(
        edge_probability, tensors, len(outputs["switch_logit"])
    )
    vertex_iou = soft_iou_loss(vertex_probability, tensors["gt_boundary"])
    contrastive = contrastive_edge_loss(
        outputs["embedding"], tensors, contrastive_max_edges, margin=0.20
    )
    total = (
        weights["switch"] * switch_loss
        + weights["aux_boundary"] * auxiliary_boundary_loss
        + weights["distance"] * distance_loss
        + weights["edge_bce"] * edge_bce
        + weights["edge_iou"] * edge_iou
        + weights["vertex_iou"] * vertex_iou
        + weights["contrastive"] * contrastive
    )
    return total, {
        "total": float(total.detach().cpu()),
        "switch": float(switch_loss.detach().cpu()),
        "aux_boundary": float(auxiliary_boundary_loss.detach().cpu()),
        "distance": float(distance_loss.detach().cpu()),
        "edge_bce": float(edge_bce.detach().cpu()),
        "edge_iou": float(edge_iou.detach().cpu()),
        "vertex_iou": float(vertex_iou.detach().cpu()),
        "contrastive": float(contrastive.detach().cpu()),
        "gate_mean": float(outputs["gate_mean"].detach().cpu()),
    }


def compute_positive_weights(cases):
    switch_positive = 0
    switch_negative = 0
    vertex_positive = 0
    vertex_negative = 0
    edge_positive = 0
    edge_negative = 0
    for case in cases:
        valid_target = case["switch_target"][case["switch_valid"]]
        switch_positive += int((valid_target > 0.5).sum())
        switch_negative += int((valid_target <= 0.5).sum())
        vertex_positive += int((case["gt_boundary_local"] > 0.5).sum())
        vertex_negative += int((case["gt_boundary_local"] <= 0.5).sum())
        edge_positive += int((case["gt_edge_boundary"] > 0.5).sum())
        edge_negative += int((case["gt_edge_boundary"] <= 0.5).sum())

    def ratio(negative, positive):
        return float(np.clip(negative / max(positive, 1), 1.0, 20.0))

    return {
        "switch": ratio(switch_negative, switch_positive),
        "vertex_boundary": ratio(vertex_negative, vertex_positive),
        "edge_boundary": ratio(edge_negative, edge_positive),
        "counts": {
            "switch_positive": switch_positive,
            "switch_negative": switch_negative,
            "vertex_positive": vertex_positive,
            "vertex_negative": vertex_negative,
            "edge_positive": edge_positive,
            "edge_negative": edge_negative,
        },
    }


def transition_matched_edges(case):
    local_edges = case["local_edges"]
    current = case["labels"][case["band_indices"]]
    alternative = case["alternative_labels"]
    low = np.minimum(current, alternative)
    high = np.maximum(current, alternative)
    return (low[local_edges[:, 0]] == low[local_edges[:, 1]]) & (
        high[local_edges[:, 0]] == high[local_edges[:, 1]]
    )


def smooth_probability(case, probability, beta, steps):
    if beta <= 0.0 or steps <= 0:
        return probability.copy()
    edges = case["local_edges"]
    matched = transition_matched_edges(case)
    edges = edges[matched]
    if not len(edges):
        return probability.copy()
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    result = probability.copy().astype(np.float32)
    degree = np.bincount(source, minlength=len(result)).astype(np.float32)
    for _ in range(steps):
        mean = np.bincount(
            source, weights=result[target], minlength=len(result)
        ).astype(np.float32) / np.maximum(degree, 1.0)
        has_neighbor = degree > 0
        result[has_neighbor] = (
            (1.0 - beta) * result[has_neighbor] + beta * mean[has_neighbor]
        )
    return result


def apply_structured_probability(
    case,
    probability,
    threshold,
    max_change_fraction,
    smoothing_beta,
    smoothing_steps,
    minimum_selected_neighbors,
):
    score = smooth_probability(case, probability, smoothing_beta, smoothing_steps)
    selected = score >= threshold
    if minimum_selected_neighbors > 0 and selected.any():
        edges = case["local_edges"]
        matched = transition_matched_edges(case)
        edges = edges[matched]
        if len(edges):
            source = np.concatenate((edges[:, 0], edges[:, 1]))
            target = np.concatenate((edges[:, 1], edges[:, 0]))
            support = np.bincount(
                source,
                weights=selected[target].astype(np.float32),
                minlength=len(selected),
            )
            selected &= support >= minimum_selected_neighbors
    selected_local = np.flatnonzero(selected)
    cap = max(1, int(round(max_change_fraction * len(case["labels"]))))
    if len(selected_local) > cap:
        selected_local = selected_local[np.argsort(score[selected_local])[-cap:]]
    labels = case["labels"].copy()
    global_indices = case["band_indices"][selected_local]
    labels[global_indices] = case["alternative_labels"][selected_local]
    return labels, int(len(global_indices)), score


def predict_probabilities(model, cases, device, boundary_feature_count):
    model.eval()
    probabilities = []
    diagnostics = []
    with torch.no_grad():
        for case in cases:
            tensors = case_to_tensors(case, device, boundary_feature_count)
            outputs = model(
                tensors["boundary_x"],
                tensors["semantic_x"],
                tensors["source"],
                tensors["target"],
            )
            probabilities.append(
                torch.sigmoid(outputs["switch_logit"]).detach().cpu().numpy()
            )
            diagnostics.append(
                {
                    "scan_id": case["stem"],
                    "probability_mean": float(
                        torch.sigmoid(outputs["switch_logit"]).mean().cpu()
                    ),
                    "probability_max": float(
                        torch.sigmoid(outputs["switch_logit"]).max().cpu()
                    ),
                    "gate_mean": float(outputs["gate_mean"].cpu()),
                }
            )
    return probabilities, diagnostics


def select_validation_config(
    cases,
    probabilities,
    iou_guard_drop,
    thresholds,
    caps,
    smoothing_betas,
    smoothing_steps,
    support_values,
):
    baseline = aggregate_metrics(cases, [case["labels"] for case in cases])
    table = []
    for beta in smoothing_betas:
        for support in support_values:
            for threshold in thresholds:
                for cap in caps:
                    predictions = []
                    changes = []
                    for case, probability in zip(cases, probabilities):
                        prediction, changed, _ = apply_structured_probability(
                            case,
                            probability,
                            threshold,
                            cap,
                            beta,
                            smoothing_steps,
                            support,
                        )
                        predictions.append(prediction)
                        changes.append(changed)
                    metrics = aggregate_metrics(cases, predictions)
                    table.append(
                        {
                            "threshold": threshold,
                            "max_change_fraction": cap,
                            "smoothing_beta": beta,
                            "smoothing_steps": smoothing_steps,
                            "minimum_selected_neighbors": support,
                            "BIoU": metrics["BIoU"],
                            "IoU": metrics["IoU"],
                            "mean_changed_vertices": float(np.mean(changes)),
                            "total_changed_vertices": int(np.sum(changes)),
                            "passes_iou_guard": metrics["IoU"]
                            >= baseline["IoU"] - iou_guard_drop,
                        }
                    )
    eligible = [row for row in table if row["passes_iou_guard"]]
    if not eligible:
        raise RuntimeError("No validation configuration passes IoU guard")
    selected = max(
        eligible,
        key=lambda row: (row["BIoU"], row["IoU"], -row["total_changed_vertices"]),
    )
    return baseline, selected, table


def write_predictions(cases, probabilities, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for case, probability in zip(cases, probabilities):
        refined, changed, smoothed = apply_structured_probability(
            case,
            probability,
            config["threshold"],
            config["max_change_fraction"],
            config["smoothing_beta"],
            config["smoothing_steps"],
            config["minimum_selected_neighbors"],
        )
        output = dict(case["prediction"])
        output["labels"] = refined.astype(int).tolist()
        if "instances" in output:
            output["instances"] = refined.astype(int).tolist()
        output_path = output_dir / (case["stem"] + ".json")
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        score_path = output_dir / (case["stem"] + "_switch_probability.npz")
        np.savez_compressed(
            score_path,
            band_indices=case["band_indices"],
            raw_probability=probability.astype(np.float32),
            structured_probability=smoothed.astype(np.float32),
        )
        records.append(
            {
                "scan_id": case["stem"],
                "changed_vertices": changed,
                "output": str(output_path),
                "output_sha256": sha256(output_path),
                "score_output_sha256": sha256(score_path),
                "prediction_input_sha256": sha256(case["prediction_path"]),
                "score_input_sha256": sha256(case["score_path"]),
                "obj_sha256": sha256(case["obj_path"]),
            }
        )
    return records


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--test-pred-root", type=Path)
    parser.add_argument("--train-score-root", type=Path, required=True)
    parser.add_argument("--val-score-root", type=Path, required=True)
    parser.add_argument("--test-score-root", type=Path)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--emit-test", action="store_true")
    parser.add_argument("--band-hops", type=int, default=2)
    parser.add_argument("--boundary-distance-hops", type=int, default=8)
    parser.add_argument("--boundary-feature-count", type=int, default=13)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--aux-boundary-weight", type=float, default=0.20)
    parser.add_argument("--distance-weight", type=float, default=0.10)
    parser.add_argument("--edge-bce-weight", type=float, default=0.50)
    parser.add_argument("--edge-iou-weight", type=float, default=0.75)
    parser.add_argument("--vertex-iou-weight", type=float, default=0.75)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-max-edges", type=int, default=30000)
    parser.add_argument("--iou-guard-drop", type=float, default=0.001)
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--caps", default="0.0005,0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--smoothing-betas", default="0.0,0.15,0.30")
    parser.add_argument("--smoothing-steps", type=int, default=1)
    parser.add_argument("--minimum-selected-neighbors", default="0,1")
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.run_dir.exists():
        raise FileExistsError("Run exists: {}".format(args.run_dir))
    if args.emit_test and (args.test_pred_root is None or args.test_score_root is None):
        raise ValueError("--emit-test requires test prediction and score roots")
    args.run_dir.mkdir(parents=True)
    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_cases = load_cases_limited(
        args.train_pred_roots,
        args.train_score_root,
        args.obj_root,
        args.json_root,
        args.band_hops,
        True,
        args.max_train_cases,
        args.boundary_distance_hops,
    )
    val_cases = load_cases_limited(
        [args.val_pred_root],
        args.val_score_root,
        args.obj_root,
        args.json_root,
        args.band_hops,
        True,
        args.max_val_cases,
        args.boundary_distance_hops,
    )
    mean, std = fit_standardizer(train_cases)
    apply_standardizer(train_cases, mean, std)
    apply_standardizer(val_cases, mean, std)
    np.savez_compressed(args.run_dir / "feature_standardizer.npz", mean=mean, std=std)

    positive_weights = compute_positive_weights(train_cases)
    model = AmbiguityGatedGraphRefiner(
        boundary_dim=args.boundary_feature_count + train_cases[0]["geometry"].shape[1],
        semantic_dim=train_cases[0]["features"].shape[1] - args.boundary_feature_count,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    loss_weights = {
        "switch": args.switch_weight,
        "aux_boundary": args.aux_boundary_weight,
        "distance": args.distance_weight,
        "edge_bce": args.edge_bce_weight,
        "edge_iou": args.edge_iou_weight,
        "vertex_iou": args.vertex_iou_weight,
        "contrastive": args.contrastive_weight,
    }
    thresholds = parse_float_list(args.thresholds)
    caps = parse_float_list(args.caps)
    smoothing_betas = parse_float_list(args.smoothing_betas)
    support_values = parse_int_list(args.minimum_selected_neighbors)

    baseline = aggregate_metrics(val_cases, [case["labels"] for case in val_cases])
    oracle_predictions = []
    for case in val_cases:
        oracle = case["labels"].copy()
        indices = case["band_indices"]
        gt_band = case["ground_truth"][indices]
        correctable = (
            (gt_band == case["alternative_labels"])
            & (gt_band != oracle[indices])
        )
        oracle[indices[correctable]] = case["alternative_labels"][correctable]
        oracle_predictions.append(oracle)
    oracle_metrics = aggregate_metrics(val_cases, oracle_predictions)

    history = []
    best = None
    best_path = args.run_dir / "best_model.pt"
    order_rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        for case_index in order_rng.permutation(len(train_cases)):
            case = train_cases[int(case_index)]
            print(
                "TRAIN_GRAPH_CASE epoch={} scan={} band={} local_edges={}".format(
                    epoch, case["stem"], len(case["band_indices"]), len(case["local_edges"])
                ),
                flush=True,
            )
            tensors = case_to_tensors(case, device, args.boundary_feature_count)
            optimizer.zero_grad()
            outputs = model(
                tensors["boundary_x"],
                tensors["semantic_x"],
                tensors["source"],
                tensors["target"],
            )
            loss, loss_row = calculate_loss(
                outputs,
                tensors,
                loss_weights,
                positive_weights,
                args.contrastive_max_edges,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss on {}".format(case["stem"]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            rows.append(loss_row)
        scheduler.step()

        val_probabilities, val_diagnostics = predict_probabilities(
            model, val_cases, device, args.boundary_feature_count
        )
        _, selected, selection_table = select_validation_config(
            val_cases,
            val_probabilities,
            args.iou_guard_drop,
            thresholds,
            caps,
            smoothing_betas,
            args.smoothing_steps,
            support_values,
        )
        mean_losses = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        epoch_row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": mean_losses,
            "validation_selected": selected,
            "validation_relative_biou_gain": selected["BIoU"] / baseline["BIoU"] - 1.0,
            "validation_diagnostics": val_diagnostics,
        }
        history.append(epoch_row)
        print(
            "GRAPH_REFINER_EPOCH {}/{} loss={:.6f} val_biou={:.9f} baseline={:.9f} rel={:.6f} iou={:.9f} threshold={} cap={} beta={} support={}".format(
                epoch,
                args.epochs,
                mean_losses["total"],
                selected["BIoU"],
                baseline["BIoU"],
                selected["BIoU"] / baseline["BIoU"] - 1.0,
                selected["IoU"],
                selected["threshold"],
                selected["max_change_fraction"],
                selected["smoothing_beta"],
                selected["minimum_selected_neighbors"],
            ),
            flush=True,
        )
        if best is None or (
            selected["BIoU"], selected["IoU"], -selected["total_changed_vertices"]
        ) > (
            best["selected"]["BIoU"],
            best["selected"]["IoU"],
            -best["selected"]["total_changed_vertices"],
        ):
            best = {
                "epoch": epoch,
                "selected": selected,
                "selection_table": selection_table,
            }
            torch.save(
                {
                    "schema": SCHEMA,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "boundary_dim": args.boundary_feature_count
                        + train_cases[0]["geometry"].shape[1],
                        "semantic_dim": train_cases[0]["features"].shape[1]
                        - args.boundary_feature_count,
                        "hidden_dim": args.hidden_dim,
                        "graph_layers": args.graph_layers,
                        "dropout": args.dropout,
                    },
                    "selected": selected,
                    "feature_mean": mean,
                    "feature_std": std,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_probabilities, final_diagnostics = predict_probabilities(
        model, val_cases, device, args.boundary_feature_count
    )
    val_records = write_predictions(
        val_cases,
        val_probabilities,
        best["selected"],
        args.run_dir / "predictions" / "val",
    )

    test_records = []
    if args.emit_test:
        test_cases = load_cases_limited(
            [args.test_pred_root],
            args.test_score_root,
            args.obj_root,
            args.json_root,
            args.band_hops,
            False,
            0,
            args.boundary_distance_hops,
        )
        apply_standardizer(test_cases, mean, std)
        test_probabilities, _ = predict_probabilities(
            model, test_cases, device, args.boundary_feature_count
        )
        test_records = write_predictions(
            test_cases,
            test_probabilities,
            best["selected"],
            args.run_dir / "predictions" / "test",
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_hashes = {
        name: tensor_sha256(tensor) for name, tensor in model.state_dict().items()
    }
    result = {
        "status": "PASS",
        "schema": SCHEMA,
        "scope": "frozen_tgn_plus_frozen_2d_full_resolution_boundary_graph_refiner",
        "primary_metric": "original_one_ring_zero_tolerance_BIoU",
        "test_ground_truth_read_by_training_script": False,
        "test_predictions_emitted": bool(args.emit_test),
        "arguments": {
            key: (
                [str(item) for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "features": {
            "base_feature_names": feature_names(),
            "geometry_feature_names": [
                "normal_x",
                "normal_y",
                "normal_z",
                "curvature_mean",
                "curvature_max",
                "coarse_boundary_hop_distance",
                "2d_boundary_ambiguity",
            ],
            "boundary_feature_count": args.boundary_feature_count,
            "score_keys": list(SCORE_KEYS),
            "standardizer_sha256": sha256(args.run_dir / "feature_standardizer.npz"),
        },
        "training": {
            "scan_count": len(train_cases),
            "positive_weights": positive_weights,
            "loss_weights": loss_weights,
            "parameter_count": parameter_count,
            "history": history,
        },
        "validation": {
            "scan_count": len(val_cases),
            "baseline": baseline,
            "oracle_candidate_upper_bound": oracle_metrics,
            "best_epoch": best["epoch"],
            "selected": best["selected"],
            "relative_biou_gain": best["selected"]["BIoU"] / baseline["BIoU"] - 1.0,
            "selection_table": best["selection_table"],
            "diagnostics": final_diagnostics,
        },
        "model": {
            "path": str(best_path),
            "sha256": sha256(best_path),
            "state_tensor_sha256": model_hashes,
        },
        "outputs": {"validation": val_records, "test": test_records},
        "duration_seconds": time.time() - started,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    result_path = args.run_dir / "training_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.run_dir / "COMPLETE").write_text(
        "status=PASS\nresult_sha256={}\n".format(sha256(result_path)), encoding="utf-8"
    )
    print(
        "GRAPH_REFINER_STATUS PASS best_epoch={} val_biou={:.9f} baseline={:.9f} relative_gain={:.6f} val_iou={:.9f} model_sha256={} result_sha256={}".format(
            best["epoch"],
            best["selected"]["BIoU"],
            baseline["BIoU"],
            best["selected"]["BIoU"] / baseline["BIoU"] - 1.0,
            best["selected"]["IoU"],
            sha256(best_path),
            sha256(result_path),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
