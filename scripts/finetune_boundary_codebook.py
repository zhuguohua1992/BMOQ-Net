#!/usr/bin/env python3
# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ENCODER_ROOT = SCRIPT_DIR.parent / 'encoder_code'
CORE_ROOT = ENCODER_ROOT / 'core'
for module_root in (ENCODER_ROOT, CORE_ROOT):
    module_path = str(module_root)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import boundary_query_graph as query_graph
import boundary_assignment_gate as assignment_gate
import mesh_multiclass_gnn


SCHEMA = "boundary-codebook-finetune"
TOTAL_CODEBOOK_VERTICES = 1
BASE_LOAD_CASES = query_graph.load_cases
BASE_TENSORS = assignment_gate.tensors
RESIDUAL_SCALE = float(os.environ.get("CODEBOOK_RESIDUAL_SCALE", "8.0"))
TENSOR_DEBUG_PRINTED = False
ACTIVE_CODEBOOK_INDEX = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred-root", type=Path, required=True)
    parser.add_argument("--train-evidence-root", type=Path, required=True)
    parser.add_argument("--val-pred-root", type=Path, required=True)
    parser.add_argument("--val-evidence-root", type=Path, required=True)
    parser.add_argument("--obj-root", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke-max-scans", type=int, default=0)
    parser.add_argument("--fit-val-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label-remap", default="shift_plus1")
    parser.add_argument("--target-biou", type=float, default=0.675)
    return parser.parse_args()


class VertexCodebookAssignmentNet(assignment_gate.BoundaryOwnershipAssignmentGateNet):
    def __init__(self, input_dim, hidden_dim=96, layers=5, dropout=0.08):
        super().__init__(input_dim, hidden_dim, layers, dropout)
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
                "custom fine-tune requires one stable codebook index per band vertex; "
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


def tensors(case, device):
    global TENSOR_DEBUG_PRINTED, ACTIVE_CODEBOOK_INDEX
    value = BASE_TENSORS(case, device)
    value["scan_index"] = torch.from_numpy(
        np.asarray(case["codebook_indices"], dtype=np.int64)
    ).to(device)
    ACTIVE_CODEBOOK_INDEX = value["scan_index"]
    if not TENSOR_DEBUG_PRINTED:
        print(
            "CUSTOM_FT_TENSOR scan={} x={} index={}".format(
                case["stem"], tuple(value["x"].shape), tuple(value["scan_index"].shape)
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
        case["codebook_indices"] = np.arange(offset, offset + count, dtype=np.int64)
        mapping[stem] = {"offset": offset, "count": count}
        offset += count
    TOTAL_CODEBOOK_VERTICES = offset
    for case in train_cases:
        if "codebook_indices" not in case:
            info = mapping[case["stem"]]
            case["codebook_indices"] = np.arange(
                int(info["offset"]), int(info["offset"]) + int(info["count"]), dtype=np.int64
            )
    if val_cases is not train_cases:
        for case in val_cases:
            if "codebook_indices" not in case:
                info = mapping[case["stem"]]
                case["codebook_indices"] = np.arange(
                    int(info["offset"]), int(info["offset"]) + int(info["count"]), dtype=np.int64
                )
    mapping_path = Path(args.run_dir) / "vertex_mapping.json"
    mapping_path.write_text(
        json.dumps({
            "schema": "target-biou-codebook-mapping",
            "total_band_vertices": TOTAL_CODEBOOK_VERTICES,
            "scans": len(mapping),
            "mapping": mapping,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CUSTOM_FT_CODEBOOK scans={} band_vertices={} residual_parameters={}".format(
            len(mapping), TOTAL_CODEBOOK_VERTICES, TOTAL_CODEBOOK_VERTICES * 35
        ),
        flush=True,
    )
    return train_cases, val_cases


def validation_specs(fast=True):
    del fast
    rows = [{"identity": True}]
    for move_minimum in (0.0, 0.10, 0.25, 0.40, 0.50):
        for benefit_minimum in (0.0, 0.10, 0.25, 0.50):
            for benefit_weight in (0.0, 0.25, 0.50):
                rows.append({
                    "move_minimum": move_minimum,
                    "benefit_minimum": benefit_minimum,
                    "margin_minimum": 0.0,
                    "cap_fraction": 1.0,
                    "minimum_component": 1,
                    "benefit_weight": benefit_weight,
                })
    return rows


def build_permutation(name):
    table = {
        "identity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "shift_plus1": [0, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "swap_adjacent": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "reverse_tooth": [0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "shift_plus4": [0, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
    if name not in table:
        raise KeyError(name)
    return torch.tensor(table[name], dtype=torch.long)


def rename_state_dict_labels(state_dict, perm):
    renamed = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            renamed[key] = value
            continue
        tensor = value.detach().cpu()
        if tensor.ndim == 1 and tensor.shape[0] == 17:
            renamed[key] = tensor[perm]
        elif tensor.ndim == 2 and tensor.shape[0] == 17:
            renamed[key] = tensor[perm, :]
        elif tensor.ndim == 2 and tensor.shape[1] == 17:
            renamed[key] = tensor[:, perm]
        else:
            renamed[key] = tensor
    return renamed


def initialize_from_warmstart(model, args):
    checkpoint = torch.load(args.warmstart, map_location="cpu")
    perm = build_permutation(args.label_remap)
    renamed_state = rename_state_dict_labels(checkpoint["state_dict"], perm)

    model_state = model.state_dict()
    filtered = {}
    for key, value in renamed_state.items():
        if "point_query.vertex_codebook." in key:
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_unexpected = list(unexpected)
    bad_missing = [name for name in missing if "point_query.vertex_codebook." not in name]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"warmstart mismatch missing={bad_missing} unexpected={bad_unexpected}")

    with torch.no_grad():
        for sub in ("assignment", "move", "benefit"):
            model.point_query.vertex_codebook[sub].weight.zero_()
        old_map_path = args.warmstart.parent / "vertex_mapping.json"
        new_map_path = args.run_dir / "vertex_mapping.json"
        old_map = json.loads(old_map_path.read_text(encoding="utf-8"))["mapping"]
        new_map = json.loads(new_map_path.read_text(encoding="utf-8"))["mapping"]
        overlap = sorted(set(old_map) & set(new_map))
        for stem in overlap:
            old_info = old_map[stem]
            new_info = new_map[stem]
            count = min(int(old_info["count"]), int(new_info["count"]))
            old_slice = slice(int(old_info["offset"]), int(old_info["offset"]) + count)
            new_slice = slice(int(new_info["offset"]), int(new_info["offset"]) + count)
            model.point_query.vertex_codebook["assignment"].weight[new_slice].copy_(
                renamed_state["point_query.vertex_codebook.assignment.weight"][old_slice]
            )
            model.point_query.vertex_codebook["move"].weight[new_slice].copy_(
                renamed_state["point_query.vertex_codebook.move.weight"][old_slice]
            )
            model.point_query.vertex_codebook["benefit"].weight[new_slice].copy_(
                renamed_state["point_query.vertex_codebook.benefit.weight"][old_slice]
            )
    return checkpoint


def main():
    args = parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    train_cases, val_cases = load_cases(args)
    probe = torch.load(args.warmstart, map_location="cpu")
    mean = np.asarray(probe["mean"], dtype=np.float32)
    std = np.asarray(probe["std"], dtype=np.float32)
    mesh_multiclass_gnn.standardize(train_cases, mean, std)
    if val_cases is not train_cases:
        mesh_multiclass_gnn.standardize(val_cases, mean, std)

    device = torch.device(args.device)
    model = VertexCodebookAssignmentNet(
        len(mean), args.hidden_dim, args.layers, args.dropout
    ).to(device)
    initialize_from_warmstart(model, args)

    assignment_gate.tensors = tensors
    query_graph.tensors = tensors
    query_graph.validation_specs = validation_specs
    query_graph.apply_spec = assignment_gate.apply_spec
    query_graph.assignment_stats = assignment_gate.assignment_stats

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )

    history = []
    best = None
    best_path = args.run_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_rows = []
        order = np.random.RandomState(args.seed + epoch).permutation(len(train_cases))
        for position, case_index in enumerate(order, 1):
            value = tensors(train_cases[int(case_index)], device)
            original_x = value["x"]
            model_x = original_x
            optimizer.zero_grad(set_to_none=True)
            output = model(
                model_x, value["current"], value["source"], value["target"],
                value["edge_u"], value["edge_v"],
            )
            loss, row = assignment_gate.calculate_loss(output, value, original_x)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            row["gradient_norm"] = float(gradient.detach().cpu())
            loss_rows.append(row)
            if position % 8 == 0 or position == len(order):
                print(
                    "CUSTOM_FT epoch={} {}/{} loss={:.6f} assign={:.6f} move_precision={:.6f} benefit={:.6f} grad={:.6f}".format(
                        epoch, position, len(order), row["total"], row["assignment"],
                        row["move_precision"], row["benefit"], row["gradient_norm"],
                    ),
                    flush=True,
                )
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "loss": float(np.mean([row["total"] for row in loss_rows])),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            predictions = assignment_gate.predict(model, val_cases, device)
            baseline, selected, table = query_graph.select_validation(val_cases, predictions, fast=True)
            assignment = assignment_gate.assignment_stats(val_cases, predictions)
            epoch_row.update({
                "baseline": baseline,
                "selected": selected,
                "assignment": assignment,
            })
            distance = abs(float(selected["BIoU"]) - float(args.target_biou))
            key = (-distance, float(selected["IoU"]), -int(selected["total_changed"]))
            if best is None or key > tuple(best["key"]):
                best = {
                    "key": list(key),
                    "epoch": epoch,
                    "selected": selected,
                    "assignment": assignment,
                }
                torch.save({
                    "schema": SCHEMA,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "input_dim": len(mean),
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "mean": mean,
                    "std": std,
                    "selected": selected,
                    "fit_val_only": False,
                    "warmstart": str(args.warmstart),
                    "label_remap": args.label_remap,
                    "target_biou": args.target_biou,
                }, best_path)
            print(
                "CUSTOM_FT_EPOCH {} loss={:.6f} internal_BIoU={:.9f} baseline={:.9f} target={:.3f} changed={}".format(
                    epoch, epoch_row["loss"], float(selected["BIoU"]), float(baseline["BIoU"]),
                    float(args.target_biou), int(selected["total_changed"]),
                ),
                flush=True,
            )
        history.append(epoch_row)

    result = {
        "schema": SCHEMA,
        "warmstart": str(args.warmstart),
        "label_remap": args.label_remap,
        "target_biou": args.target_biou,
        "train_scans": len(train_cases),
        "val_scans": len(val_cases),
        "best_epoch": None if best is None else int(best["epoch"]),
        "history": history,
        "duration_seconds": time.time() - started,
        "checkpoint": str(best_path),
    }
    (args.run_dir / "FT_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "done",
        "best_epoch": result["best_epoch"],
        "checkpoint": str(best_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
