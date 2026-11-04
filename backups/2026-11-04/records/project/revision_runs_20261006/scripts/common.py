"""Paths, atomic output and provenance shared by the independent revision queue."""
from pathlib import Path
import csv
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone, timedelta

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
PYTHON = Path('/home/renguanming/.conda/envs/beta_fft/bin/python')
DATA = Path('/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC')
FULL = PROJECT / 'runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT'
BASE = PROJECT / 'runs/beta_fft_acdc_official_full_seed42_fresh_ckpt'

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text())

def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def read_table(path):
    path = Path(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))

def bootstrap():
    sys.path.insert(0, str(WORK / 'source/train_ACDC'))
    sys.path.insert(0, str(WORK / 'source'))

def config_for(task):
    paths = sorted((WORK / 'configs').glob(task + '_*.yaml'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one config for {task}: {paths}')
    # JSON is a YAML 1.2 subset; avoids ambiguous scalar coercion.
    return paths[0], read_json(paths[0])

def analysis_status(task, status, reason, **kwargs):
    atomic_json(WORK / f'analysis/{task}/status.json', dict(task=task, status=status, reason=reason, time=now(), **kwargs))
