#!/usr/bin/env python3
# 本文件用于实现完整模型的边界编码、训练或推理。
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ENCODER_ROOT = SCRIPT_DIR.parent / 'encoder_code'
CORE_ROOT = ENCODER_ROOT / 'core'
for module_root in (ENCODER_ROOT, CORE_ROOT):
    module_path = str(module_root)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import boundary_query_graph as query_graph
import boundary_assignment_gate as assignment_gate
import boundary_codebook_train as codebook_trainer
import mesh_multiclass_gnn


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-root', type=Path, required=True)
    ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--obj-root', type=Path, required=True)
    ap.add_argument('--json-root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--mapping-json', type=Path, default=None)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--device', default='cuda')
    return ap.parse_args()


def load_subset_cases(args):
    loader_args = SimpleNamespace(
        val_pred_root=args.pred_root,
        val_evidence_root=args.evidence_root,
        obj_root=args.obj_root,
        json_root=args.json_root,
        workers=args.workers,
        smoke_max_scans=0,
        train_pred_root=None,
        train_evidence_root=None,
        run_dir=args.output_root,
    )
    _, cases = codebook_trainer.BASE_LOAD_CASES(loader_args)
    return cases


def assign_large_mapping(cases, mapping_payload, checkpoint):
    mapping = mapping_payload['mapping']
    total_from_mapping = int(mapping_payload.get('total_band_vertices', 0))
    total_from_ckpt = int(checkpoint['state_dict']['point_query.vertex_codebook.assignment.weight'].shape[0])
    codebook_trainer.TOTAL_CODEBOOK_VERTICES = max(total_from_mapping, total_from_ckpt)
    for case in cases:
        stem = case['stem']
        if stem not in mapping:
            raise KeyError(f'missing stem in mapping: {stem}')
        info = mapping[stem]
        count = int(info['count'])
        offset = int(info['offset'])
        band_count = int(len(case['band_indices']))
        if count != band_count:
            raise ValueError(f'band count mismatch for {stem}: mapping={count} case={band_count}')
        case['codebook_indices'] = list(range(offset, offset + count))


def main():
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    mapping_json = args.mapping_json or (args.checkpoint.parent / 'vertex_mapping.json')
    mapping_payload = json.loads(mapping_json.read_text(encoding='utf-8'))

    cases = load_subset_cases(args)
    assign_large_mapping(cases, mapping_payload, checkpoint)
    mesh_multiclass_gnn.standardize(cases, checkpoint['mean'], checkpoint['std'])

    query_graph.tensors = codebook_trainer.tensors
    assignment_gate.tensors = codebook_trainer.tensors
    query_graph.apply_spec = assignment_gate.apply_spec

    device = torch.device(args.device)
    model = codebook_trainer.VertexCodebookAssignmentNet(
        int(checkpoint.get('input_dim', len(checkpoint['mean']))),
        int(checkpoint.get('hidden_dim', 96)),
        int(checkpoint.get('layers', 5)),
        float(checkpoint.get('dropout', 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    predictions = assignment_gate.predict(model, cases, device)
    selected = dict(checkpoint['selected'])

    prediction_root = args.output_root / 'predictions'
    prediction_root.mkdir()
    rows = []
    for case, prediction in zip(cases, predictions):
        labels, instances, changed = assignment_gate.apply_spec(case, prediction, selected)
        payload = dict(case['payload'])
        payload['labels'] = labels.astype(int).tolist()
        payload['instances'] = instances.astype(int).tolist()
        path = prediction_root / f"{case['stem']}.json"
        path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
        rows.append({'scan_id': case['stem'], 'changed': int(changed), 'sha256': digest(path)})

    result = {
        'schema': 'boundary-codebook-materialization-subset-mapping',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': digest(args.checkpoint),
        'mapping_json': str(mapping_json),
        'mapping_sha256': digest(mapping_json),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'selected': selected,
        'prediction_count': len(rows),
        'total_changed': int(sum(r['changed'] for r in rows)),
        'rows': rows,
    }
    (args.output_root / 'materialization_result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'checkpoint': str(args.checkpoint),
        'prediction_count': result['prediction_count'],
        'total_changed': result['total_changed'],
        'selected_BIoU': selected.get('BIoU'),
        'selected_IoU': selected.get('IoU'),
        'large_codebook_vertices': codebook_trainer.TOTAL_CODEBOOK_VERTICES,
    }, indent=2))


if __name__ == '__main__':
    main()
