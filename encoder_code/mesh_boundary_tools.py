# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_obj(path):
    vertex_count = 0
    faces = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split()
            if not fields:
                continue
            if fields[0] == "v":
                vertex_count += 1
            elif fields[0] == "f" and len(fields) >= 4:
                indices = []
                for token in fields[1:]:
                    index = int(token.split("/", 1)[0])
                    index = vertex_count + index if index < 0 else index - 1
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    return vertex_count, np.asarray(faces, dtype=np.int64)


def build_neighbors(vertex_count, triangles):
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.fromiter(sorted(values), dtype=np.int64) for values in neighbors]


def boundary_mask(labels, neighbors):
    result = np.zeros(len(labels), dtype=np.bool_)
    for index, adjacent in enumerate(neighbors):
        if len(adjacent) and np.any(labels[adjacent] != labels[index]):
            result[index] = True
    return result


def fdi_to_internal(label, jaw):
    label = int(label)
    if label == 0:
        return 0
    if jaw == "lower":
        label -= 20
    quadrant, tooth = divmod(label, 10)
    if quadrant == 1 and 1 <= tooth <= 8:
        return tooth
    if quadrant == 2 and 1 <= tooth <= 8:
        return tooth + 8
    raise ValueError("Unsupported FDI {} for {}".format(label, jaw))


def modal(values):
    names, counts = np.unique(values, return_counts=True)
    return int(names[int(np.argmax(counts))])


def expand_band(mask, neighbors, rings):
    band = mask.copy()
    frontier = np.flatnonzero(mask)
    for _ in range(max(0, int(rings) - 1)):
        added = []
        for index in frontier:
            for neighbor in neighbors[int(index)]:
                if not band[int(neighbor)]:
                    band[int(neighbor)] = True
                    added.append(int(neighbor))
        frontier = np.asarray(added, dtype=np.int64)
        if not len(frontier):
            break
    return band


def boundary_nll(index, state, neighbors, boundary_probability, override_index, override_value):
    value = override_value if index == override_index else state[index]
    adjacent = neighbors[index]
    is_boundary = False
    for neighbor in adjacent:
        neighbor = int(neighbor)
        other = override_value if neighbor == override_index else state[neighbor]
        if value != other:
            is_boundary = True
            break
    probability = float(np.clip(boundary_probability[index], 1.0e-4, 1.0 - 1.0e-4))
    return -math.log(probability if is_boundary else 1.0 - probability)


def refine(payload, evidence, triangles, config):
    original_instances = np.asarray(payload["instances"], dtype=np.int64)
    original_labels = np.asarray(payload["labels"], dtype=np.int64)
    if original_instances.shape != original_labels.shape:
        raise ValueError("labels/instances mismatch")
    neighbors = build_neighbors(len(original_instances), triangles)
    probability = evidence["semantic_probability"].astype(np.float64)
    reliability = evidence["reliability"].astype(np.float64)
    boundary_probability = evidence["boundary_probability"].astype(np.float64)
    if probability.shape != (len(original_instances), 17):
        raise ValueError("semantic evidence shape mismatch")

    instance_to_fdi = {}
    instance_to_class = {}
    for instance_id in sorted(int(value) for value in np.unique(original_instances)):
        mask = original_instances == instance_id
        fdi = 0 if instance_id == 0 else modal(original_labels[mask])
        instance_to_fdi[instance_id] = fdi
        instance_to_class[instance_id] = fdi_to_internal(fdi, payload["jaw"])

    state = original_instances.copy()
    initial_boundary = boundary_mask(state, neighbors)
    band = expand_band(initial_boundary, neighbors, int(config["band_rings"]))
    band_indices = np.flatnonzero(band)
    changed_vertices = set()
    iterations = []
    sem_weight = float(config["semantic_weight"])
    boundary_weight = float(config["boundary_weight"])
    smooth_weight = float(config.get("smooth_weight", 0.0))
    stay_penalty = float(config["stay_penalty"])
    min_margin = float(config["minimum_semantic_margin"])
    min_reliability = float(config["minimum_reliability"])
    min_target_neighbors = int(config["minimum_target_neighbors"])
    min_target_probability = float(config.get("minimum_target_probability", 0.0))
    max_target_probability = float(config.get("maximum_target_probability", 1.0))
    direction_rules = config.get("direction_rules", {})
    min_target_fraction = float(config.get("minimum_target_neighbor_fraction", 0.0))
    max_boundary_nll_delta = float(config.get("maximum_boundary_nll_delta", float("inf")))
    min_component_size = int(config.get("minimum_proposal_component_size", 1))
    max_components = int(config.get("maximum_components_per_scan", 1000000))
    min_gain = float(config.get("minimum_energy_gain", 1.0e-8))

    for iteration in range(int(config["iterations"])):
        proposals = []
        for index in band_indices:
            index = int(index)
            adjacent = neighbors[index]
            if not len(adjacent):
                continue
            current = int(state[index])
            candidate_values, candidate_counts = np.unique(
                state[adjacent], return_counts=True
            )
            candidates = [current]
            for value, count in zip(candidate_values, candidate_counts):
                value = int(value)
                if (
                    value != current
                    and int(count) >= min_target_neighbors
                    and float(count) / float(max(1, len(adjacent))) >= min_target_fraction
                ):
                    candidates.append(value)
            if len(candidates) == 1 or reliability[index] < min_reliability:
                continue

            affected = np.concatenate((np.asarray([index], dtype=np.int64), adjacent))
            scores = {}
            current_class = instance_to_class[current]
            for candidate in candidates:
                candidate_class = instance_to_class[int(candidate)]
                if candidate != current:
                    if current == 0 and candidate != 0:
                        direction = "background_to_tooth"
                    elif current != 0 and candidate == 0:
                        direction = "tooth_to_background"
                    else:
                        direction = "tooth_to_tooth"
                    direction_rule = direction_rules.get(direction, {})
                    candidate_probability = float(probability[index, candidate_class])
                    candidate_min_probability = float(
                        direction_rule.get("minimum_target_probability", min_target_probability)
                    )
                    candidate_max_probability = float(
                        direction_rule.get("maximum_target_probability", max_target_probability)
                    )
                    if not candidate_min_probability <= candidate_probability <= candidate_max_probability:
                        continue
                    margin = (
                        float(probability[index, candidate_class])
                        - float(probability[index, current_class])
                    )
                    candidate_min_margin = float(
                        direction_rule.get("minimum_semantic_margin", min_margin)
                    )
                    candidate_max_margin = float(
                        direction_rule.get("maximum_semantic_margin", float("inf"))
                    )
                    if not candidate_min_margin <= margin <= candidate_max_margin:
                        continue
                semantic_energy = -sem_weight * (0.25 + 0.75 * reliability[index]) * math.log(
                    max(float(probability[index, candidate_class]), 1.0e-8)
                )
                change_energy = stay_penalty if candidate != int(original_instances[index]) else 0.0
                boundary_energy = 0.0
                for affected_index in affected:
                    boundary_energy += boundary_nll(
                        int(affected_index), state, neighbors,
                        boundary_probability, index, int(candidate),
                    )
                disagreement_count = int(np.count_nonzero(state[adjacent] != int(candidate)))
                degree = max(1, len(adjacent))
                smooth_energy = smooth_weight * disagreement_count / float(degree)
                scores[int(candidate)] = (
                    semantic_energy
                    + boundary_weight * boundary_energy
                    + change_energy
                    + smooth_energy
                )
            if current not in scores or len(scores) == 1:
                continue
            current_boundary_energy = 0.0
            for affected_index in affected:
                current_boundary_energy += boundary_nll(
                    int(affected_index), state, neighbors,
                    boundary_probability, index, current,
                )
            allowed_scores = {current: scores[current]}
            for candidate, score in scores.items():
                if candidate == current:
                    continue
                candidate_boundary_energy = 0.0
                for affected_index in affected:
                    candidate_boundary_energy += boundary_nll(
                        int(affected_index), state, neighbors,
                        boundary_probability, index, candidate,
                    )
                if candidate_boundary_energy - current_boundary_energy <= max_boundary_nll_delta:
                    allowed_scores[candidate] = score
            scores = allowed_scores
            if len(scores) == 1:
                continue
            best = min(scores, key=lambda value: (scores[value], value))
            gain = scores[current] - scores[best]
            if best != current and gain > min_gain:
                proposals.append((gain, index, current, int(best)))

                                                                         
                                                                               
                                                                               
        by_index = {index: (gain, source, target) for gain, index, source, target in proposals}
        unvisited = set(by_index)
        components = []
        while unvisited:
            seed = min(unvisited)
            gain, source, target = by_index[seed]
            stack = [seed]
            unvisited.remove(seed)
            component = []
            while stack:
                index = stack.pop()
                component.append(index)
                for neighbor in neighbors[index]:
                    neighbor = int(neighbor)
                    if neighbor not in unvisited:
                        continue
                    _, other_source, other_target = by_index[neighbor]
                    if other_source == source and other_target == target:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
            if len(component) >= min_component_size:
                components.append({
                    "indices": component,
                    "source": source,
                    "target": target,
                    "gain": float(sum(by_index[index][0] for index in component)),
                })
        components.sort(key=lambda value: (-value["gain"], -len(value["indices"]), min(value["indices"])))
        committed = 0
        committed_components = 0
        next_state = state.copy()
        for component in components[:max_components]:
            valid = [
                index for index in component["indices"]
                if int(state[index]) == component["source"]
                and np.any(state[neighbors[index]] == component["target"])
            ]
            if len(valid) < min_component_size:
                continue
            next_state[np.asarray(valid, dtype=np.int64)] = component["target"]
            changed_vertices.update(valid)
            committed += len(valid)
            committed_components += 1
        state = next_state
        iterations.append({
            "iteration": iteration + 1,
            "proposals": len(proposals),
            "committed": committed,
            "committed_components": committed_components,
        })
        if committed == 0:
            break

    output_labels = np.asarray(
        [instance_to_fdi[int(value)] for value in state], dtype=np.int64
    )
    final_boundary = boundary_mask(state, neighbors)
    stats = {
        "band_vertices": int(band.sum()),
        "changed_vertices": int(np.count_nonzero(state != original_instances)),
        "touched_vertices": len(changed_vertices),
        "initial_boundary_vertices": int(initial_boundary.sum()),
        "final_boundary_vertices": int(final_boundary.sum()),
        "from_background": int(np.count_nonzero((original_instances == 0) & (state != 0))),
        "to_background": int(np.count_nonzero((original_instances != 0) & (state == 0))),
        "between_teeth": int(np.count_nonzero(
            (original_instances != 0) & (state != 0) & (state != original_instances)
        )),
        "iterations": iterations,
    }
    return output_labels, state, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pred-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for source_path in sorted(args.source_pred_root.glob("*.json")):
        scan_id = source_path.stem
        patient = scan_id.rsplit("_", 1)[0]
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        evidence_path = args.evidence_root / (scan_id + ".npz")
        with np.load(str(evidence_path)) as evidence_file:
            evidence = {key: evidence_file[key] for key in evidence_file.files}
        vertex_count, triangles = parse_obj(
            args.obj_root / patient / (scan_id + ".obj")
        )
        if vertex_count != len(payload["labels"]):
            raise ValueError("{} OBJ/prediction length mismatch".format(scan_id))
        labels, instances, stats = refine(payload, evidence, triangles, config)
        output = {
            "id_patient": payload.get("id_patient", ""),
            "jaw": payload["jaw"],
            "labels": labels.astype(int).tolist(),
            "instances": instances.astype(int).tolist(),
        }
        (args.output_root / (scan_id + ".json")).write_text(
            json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        stats["scan_id"] = scan_id
        receipts.append(stats)
        print("TRAIN_REFINE " + json.dumps(stats, sort_keys=True), flush=True)
    (args.output_root.parent / (args.output_root.name + "_receipt.json")).write_text(
        json.dumps({"config": config, "scans": receipts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "TRAIN_REFINE_COMPLETE scans={} changed_vertices={}".format(
            len(receipts), sum(row["changed_vertices"] for row in receipts)
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit("internal dependency; invoke launch/train_one_epoch.sh")

