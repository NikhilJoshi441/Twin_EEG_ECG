import json, os
from collections import Counter

DATA='data/fusion_raw.jsonl'
out='reports/label_dist.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
cnt=Counter()
total=0
with open(DATA,'r',encoding='utf-8') as f:
    for line in f:
        try:
            obj=json.loads(line)
            label = 1 if (obj.get('explanation') and (obj.get('explanation').get('prob') or 0) >= 0.6) else 0
            cnt[label]+=1
            total+=1
        except Exception:
            continue

report = {'total': total, 'counts': dict(cnt)}
print(report)
with open(out,'w',encoding='utf-8') as fo:
    json.dump(report, fo, indent=2)
print('Wrote', out)
