import json, os
import torch
from collections import Counter

DATA='data/fusion_balanced.jsonl'
if not os.path.exists(DATA):
    print('missing', DATA); raise SystemExit(1)
rows=[]
with open(DATA,'r',encoding='utf-8') as f:
    for line in f:
        try:
            obj=json.loads(line)
            rows.append(obj)
        except Exception:
            continue

n=len(rows)
val_n=int(n*0.2)
train_n=n-val_n
gen=torch.Generator(); gen.manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(rows, [train_n, val_n], generator=gen)

def counts(sub):
    c=Counter()
    for obj in sub:
        lab = int(obj.get('fusion_label') or 0)
        c[lab]+=1
    return dict(c)

print('total', n, 'train', len(train_ds), 'val', len(val_ds))
print('train labels', counts(train_ds))
print('val labels', counts(val_ds))
