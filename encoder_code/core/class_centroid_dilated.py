# 本文件用于实现完整模型的边界编码、训练或推理。
from pathlib import Path

import numpy as np

import edge_gated_gnn as edge_gated
import boundary_tversky_attention as boundary_tversky
import multiscale_boundary_semantic as multiscale_boundary
import boundary_contrastive_focal as boundary_contrastive
import multiview_guided_gcbl as multiview_guided
import biou_gain_ranking as biou_gain


SCHEMA = "boundary-class-centroid-dilated-exact-biou"
BASE_BUILD_CASE = multiview_guided.build_case
APPENDED_FEATURES = 72


def class_centroid_features(case, vertices):
    instances = np.asarray(case["instances"], dtype=np.int64)
    labels = np.asarray(case["labels"], dtype=np.int64)
    class_map = edge_gated.mesh_multiclass.instance_class_map(
        instances, labels, case["payload"]["jaw"]
    )
    classes = np.fromiter(
        (class_map[int(value)] for value in instances),
        dtype=np.int64,
        count=len(instances),
    )
    band_vertices = np.asarray(vertices[case["band_indices"]], dtype=np.float32)
    mesh_center = vertices.mean(axis=0)
    mesh_scale = float(np.linalg.norm(vertices - mesh_center, axis=1).max())
    mesh_scale = max(mesh_scale, 1.0e-6)

    centroids = np.repeat(mesh_center[None, :], 17, axis=0).astype(np.float32)
    radii = np.full(17, mesh_scale, dtype=np.float32)
    fractions = np.zeros(17, dtype=np.float32)
    present = np.zeros(17, dtype=np.float32)
    for class_index in range(17):
        mask = classes == class_index
        count = int(mask.sum())
        if not count:
            continue
        points = vertices[mask]
        centroid = points.mean(axis=0)
        radius = float(np.sqrt(np.square(points - centroid).sum(axis=1).mean()))
        centroids[class_index] = centroid
        radii[class_index] = max(radius, 0.01 * mesh_scale)
        fractions[class_index] = count / float(len(vertices))
        present[class_index] = 1.0

    displacement = band_vertices[:, None, :] - centroids[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    distance_mesh = np.clip(distance / mesh_scale, 0.0, 4.0)
    distance_radius = np.clip(distance / radii[None, :], 0.0, 8.0)
    repeated_fraction = np.broadcast_to(fractions[None, :], distance.shape)
    repeated_present = np.broadcast_to(present[None, :], distance.shape)
    current = np.asarray(case["current_classes"], dtype=np.int64)
    current_centroid = centroids[current]
    current_radius = radii[current]
    current_relative = (band_vertices - current_centroid) / mesh_scale
    current_radial = (
        np.linalg.norm(band_vertices - current_centroid, axis=1)
        / np.maximum(current_radius, 1.0e-6)
    )[:, None]
    return np.concatenate(
        (
            distance_mesh,
            distance_radius,
            repeated_fraction,
            repeated_present,
            current_relative,
            np.clip(current_radial, 0.0, 8.0),
        ),
        axis=1,
    ).astype(np.float32)


def build_case(*task):
    case = BASE_BUILD_CASE(*task)
    prediction_path = Path(task[0])
    obj_root = Path(task[2])
    patient = prediction_path.stem.rsplit("_", 1)[0]
    vertices, _faces = edge_gated.mesh_multiclass.parse_obj_geometry(
        obj_root / patient / (prediction_path.stem + ".obj")
    )
    added = class_centroid_features(case, vertices)
    if added.shape != (len(case["band_indices"]), APPENDED_FEATURES):
        raise ValueError(f"B unexpected centroid feature shape: {added.shape}")
    if not np.isfinite(added).all():
        raise ValueError("B non-finite centroid feature")
    case["features"] = np.concatenate((case["features"], added), axis=1).astype(np.float32)
    return case


def main():
    edge_gated.SCHEMA = SCHEMA
    edge_gated.mesh_multiclass.build_case = build_case
    edge_gated.add_gain_targets = biou_gain.exact_biou_gain_targets
    edge_gated.EdgeGatedGainAwareMeshGNN = multiscale_boundary.MultiScaleBoundarySemanticGNN
    edge_gated.calculate_loss = boundary_contrastive.calculate_loss
    edge_gated.select_validation = boundary_tversky.select_validation
    edge_gated.main()


if __name__ == "__main__":
    main()

