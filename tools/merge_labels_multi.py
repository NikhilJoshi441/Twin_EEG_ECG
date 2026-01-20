"""Merge manual labels CSV into JSONL dataset and write a labeled JSONL for training.

Assumes:
- data/fusion_raw.jsonl (or similar) exists with one JSON object per line
- data/manual_labels.csv contains `idx,label` rows mapping CSV index to integer class labels

Outputs: data/fusion_labeled.jsonl with added field `fusion_label` set to integer class index
"""
import csv, json
from pathlib import Path

RAW = Path('data/fusion_raw.jsonl')
MAN = Path('data/manual_labels.csv')
OUT = Path('data/fusion_labeled.jsonl')

if not RAW.exists():
    print('Missing raw data at', RAW); raise SystemExit(1)
if not MAN.exists():
    print('Missing manual labels at', MAN); raise SystemExit(1)

labels = {}
with open(MAN,'r',encoding='utf-8') as fh:
    r = csv.DictReader(fh)
    for row in r:
        try:
            idx = int(row.get('idx'))
            lab = row.get('label')
            # allow comma-separated for multi-label? keep simple: integer
            labels[idx] = int(lab)
        except Exception:
            continue

print('Loaded', len(labels), 'manual labels')

with open(RAW,'r',encoding='utf-8') as inf, open(OUT,'w',encoding='utf-8') as outf:
    for i,line in enumerate(inf):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if i in labels:
            obj['fusion_label'] = labels[i]
        outf.write(json.dumps(obj) + '\n')

print('Wrote labeled JSONL to', OUT)
