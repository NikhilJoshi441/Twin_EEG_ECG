import json, os, sys
from statistics import median
from collections import Counter

IN='data/fusion_raw.jsonl'
OUT='data/fusion_improved.jsonl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

if not os.path.exists(IN):
    print('input missing', IN); sys.exit(1)

rows=[]
vals=[]
with open(IN,'r',encoding='utf-8') as f:
    for line in f:
        try:
            obj=json.loads(line)
            rows.append(obj)
            avg=(obj.get('metrics') or {}).get('avg_hr_bpm')
            try: vals.append(float(avg))
            except Exception: vals.append(0.0)
        except Exception:
            continue

if not rows:
    print('no rows'); sys.exit(1)

low = sorted(vals)[max(0, int(len(vals)*0.25)-1)]
high = sorted(vals)[min(len(vals)-1, int(len(vals)*0.75))]
print('derived thresholds low,high =', low, high)

cnt = Counter()
kept=0
with open(OUT,'w',encoding='utf-8') as fo:
    for obj,v in zip(rows, vals):
        # priority: explicit fusion_label
        if 'fusion_label' in obj:
            lab = int(obj.get('fusion_label') or 0)
        else:
            prob = (obj.get('explanation') or {}).get('prob') or 0
            if prob >= 0.8:
                lab = 1
            elif prob <= 0.2:
                lab = 0
            else:
                # use avg_hr_bpm extremes for confident labels
                if v >= high:
                    lab = 1
                elif v <= low:
                    lab = 0
                else:
                    # uncertain — skip this sample
                    continue
        obj['fusion_label'] = int(lab)
        fo.write(json.dumps(obj) + '\n')
        cnt[lab]+=1
        kept+=1

print('wrote', OUT, 'kept', kept, 'counts', dict(cnt))
