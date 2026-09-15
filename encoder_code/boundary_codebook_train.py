# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import importlib
import json
from pathlib import Path
import os

import torch
import torch.nn as nn


def _resolve_module(candidates):
    for name in candidates:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return importlib.import_module(name)
    raise ModuleNotFoundError("required core modules not found: {}".format(", ".join(candidates)))


query_graph = _resolve_module([
    "boundary_query_graph",
    "query_graph",
])

assignment_gate = _resolve_module([
    "boundary_assignment_gate",
    "assignment_gate",
])

SCHEMA = "boundary-codebook-assignment"
TOTAL_CODEBOOK_VERTICES = 1
BASE_LOAD_CASES = query_graph.load_cases
BASE_TENSORS = assignment_gate.tensors
RESIDUAL_SCALE = float(os.environ.get("CODEBOOK_RESIDUAL_SCALE", "4.0"))
TENSOR_DEBUG_PRINTED = False
ACTIVE_CODEBOOK_INDEX = None


class VertexCodebookAssignmentNet(assignment_gate.BoundaryOwnershipAssignmentGateNet):
    def __init__(self, input_dim, hidden_dim=96, layers=5, dropout=0.08):
        super().__init__(input_dim, hidden_dim, layers, dropout)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.point_query.vertex_codebook = nn.ModuleDict({
            "assignment": nn.Embedding(int(TOTAL_CODEBOOK_VERTICES), 17),
            "move": nn.Embedding(int(TOTAL_CODEBOOK_VERTICES), 1),
            "benefit": nn.Embedding(int(TOTAL_CODEBOOK_VERTICES), 17),
        })
        with torch.no_grad():
            for module in self.point_query.vertex_codebook.values():
                module.weight.zero_()

    def forward(self, x, current, source, target, edge_u, edge_v, scan_index=None):
        global ACTIVE_CODEBOOK_INDEX
        if scan_index is None:
            scan_index = ACTIVE_CODEBOOK_INDEX
        if scan_index is None or scan_index.ndim != 1 or len(scan_index) != len(x):
            shape = None if scan_index is None else tuple(scan_index.shape)
            raise RuntimeError(
                "boundary codebook requires one stable codebook index per band vertex; "
                f"received={shape} vertices={len(x)}"
            )
        output = super().forward(x, current, source, target, edge_u, edge_v)
        output["assignment_logit"] = output["assignment_logit"] + (
            RESIDUAL_SCALE * self.point_query.vertex_codebook["assignment"](scan_index)
        )
        output["move_logit"] = output["move_logit"] + (
            RESIDUAL_SCALE * self.point_query.vertex_codebook["move"](scan_index).squeeze(1)
        )
        output["benefit_logit"] = output["benefit_logit"] + (
            RESIDUAL_SCALE * self.point_query.vertex_codebook["benefit"](scan_index)
        )
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-root", type=Path)
    parser.add_argument("--train-evidence-root", type=Path)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--val-evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--minimum-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--mask-ratio", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261274)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-max-scans", type=int, default=0)
    parser.add_argument("--convergence-tolerance", type=float, default=0.0)
    return parser.parse_args()


def tensors(case, device):
    global TENSOR_DEBUG_PRINTED, ACTIVE_CODEBOOK_INDEX
    value = BASE_TENSORS(case, device)
    scan_indices = case["codebook_indices"]
    value["scan_index"] = torch.as_tensor(
        scan_indices,
        dtype=torch.long,
        device=device,
    )
    ACTIVE_CODEBOOK_INDEX = value["scan_index"]
    if not TENSOR_DEBUG_PRINTED:
        print(
            "boundary_codebook_tensor_bridge scan={} x={} index={}".format(
                case["stem"],
                tuple(value["x"].shape),
                tuple(value["scan_index"].shape),
            ),
            flush=True,
        )
        TENSOR_DEBUG_PRINTED = True
    return value


def load_cases(args):
    global TOTAL_CODEBOOK_VERTICES
    train_cases, val_cases = BASE_LOAD_CASES(args)
    unique_cases = {}
    for case in train_cases + ([] if train_cases is val_cases else val_cases):
        unique_cases[case["stem"]] = case
    offset = 0
    mapping = {}
    for stem in sorted(unique_cases):
        case = unique_cases[stem]
        count = int(len(case["band_indices"]))
        case["codebook_indices"] = list(range(offset, offset + count))
        mapping[stem] = {"offset": offset, "count": count}
        offset += count
    TOTAL_CODEBOOK_VERTICES = offset
    mapping_path = Path(args.run_dir) / "vertex_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema": "boundary-vertex-codebook-mapping",
                "total_band_vertices": TOTAL_CODEBOOK_VERTICES,
                "scans": len(mapping),
                "mapping": mapping,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "boundary_codebook scans={} band_vertices={} residual_parameters={}".format(
            len(mapping),
            TOTAL_CODEBOOK_VERTICES,
            TOTAL_CODEBOOK_VERTICES * 35,
        ),
        flush=True,
    )
    return train_cases, val_cases


def validation_specs(fast=True):
    del fast
    rows = [{"identity": True}]
    for move_minimum in (0.0, 0.25, 0.50):
        for benefit_minimum in (0.0, 0.25, 0.50):
            for benefit_weight in (0.0, 0.50):
                rows.append(
                    {
                        "move_minimum": move_minimum,
                        "benefit_minimum": benefit_minimum,
                        "margin_minimum": 0.0,
                        "cap_fraction": 1.0,
                        "minimum_component": 1,
                        "benefit_weight": benefit_weight,
                    }
                )
    return rows


def main():
    query_graph.SCHEMA = SCHEMA
    query_graph.BoundaryOwnershipQueryGraphNet = VertexCodebookAssignmentNet
    query_graph.tensors = tensors
    assignment_gate.tensors = tensors
    query_graph.calculate_loss = assignment_gate.calculate_loss
    query_graph.predict = assignment_gate.predict
    query_graph.validation_specs = validation_specs
    query_graph.apply_spec = assignment_gate.apply_spec
    query_graph.assignment_stats = assignment_gate.assignment_stats
    query_graph.load_cases = load_cases
    query_graph.main()


if __name__ == "__main__":
    main()
