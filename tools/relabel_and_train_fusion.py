import json, os, math, subprocess, sys
from statistics import median

IN='data/fusion_raw.jsonl'
OUT='data/fusion_balanced.jsonl'

if not os.path.exists(IN):
    print('Input not found:', IN); sys.exit(1)

vals=[]
rows=[]
with open(IN,'r',encoding='utf-8') as f:
    for line in f:
        try:
            obj=json.loads(line)
            rows.append(obj)
            m=(obj.get('metrics') or {}).get('avg_hr_bpm')
            try:
                vals.append(float(m))
            except Exception:
                vals.append(0.0)
        except Exception:
            continue

if len(rows)==0:
    print('No rows'); sys.exit(1)

# choose threshold at median to split roughly half/half, but ensure some variability
th = float(median(vals))
print('Derived avg_hr_bpm median threshold:', th)

pos=0; neg=0
with open(OUT,'w',encoding='utf-8') as fo:
    for obj,v in zip(rows, vals):
        lab = 1 if v >= th else 0
        obj['fusion_label'] = lab
        if lab==1: pos+=1
        else: neg+=1
        fo.write(json.dumps(obj) + '\n')

print('Wrote', OUT, 'pos=',pos,'neg=',neg)

# call trainer
cmd = [sys.executable, 'tools/train_fusion.py', '--data', OUT, '--epochs', '30', '--batch', '32', '--out', 'src/models/fusion_balanced.pth']
print('Running trainer:', ' '.join(cmd))
subprocess.check_call(cmd)
