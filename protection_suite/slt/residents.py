from __future__ import annotations
from pathlib import Path
import json
import torch


def _resolve_checkpoint(manifest_path: str | Path, checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.is_absolute():
        return p
    return Path(manifest_path).resolve().parent / p


def load_resident_manifest(path: str | Path):
    path = Path(path)
    with open(path) as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or 'residents' not in obj:
        raise ValueError(f'invalid resident-bank manifest: {path}')
    rows = obj['residents']
    by_task = {str(r['task_name']): r for r in rows}
    if len(by_task) != len(rows):
        raise ValueError('resident-bank manifest has duplicate task_name entries')
    return obj, by_task


def load_resident_from_bank(world, task, manifest_path: str | Path):
    manifest, by_task = load_resident_manifest(manifest_path)
    if str(task.name) not in by_task:
        raise KeyError(f'{task.name} not found in resident bank {manifest_path}')
    rec = by_task[str(task.name)]
    ckpt_path = _resolve_checkpoint(manifest_path, rec['checkpoint'])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path, map_location=world.device, weights_only=False)
    if payload.get('dataset') != world.dataset:
        raise ValueError(f"resident bank dataset mismatch: {payload.get('dataset')} vs {world.dataset}")
    if payload.get('arch') != world.arch:
        raise ValueError(f"resident bank arch mismatch: {payload.get('arch')} vs {world.arch}")
    if int(payload.get('seed')) != int(world.seed):
        raise ValueError(f"resident bank seed mismatch: {payload.get('seed')} vs {world.seed}")
    if str(payload.get('task_name')) != str(task.name):
        raise ValueError(f"resident bank task mismatch: {payload.get('task_name')} vs {task.name}")
    model = world.model(pretrained=False)
    model.load_state_dict(payload['state_dict'])
    model.to(world.device)
    return model, rec, ckpt_path
