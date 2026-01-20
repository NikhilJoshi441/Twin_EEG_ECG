import json
from collections import Counter
p='data/fusion_labeled.jsonl'
c=Counter()
found=0
with open(p,'r',encoding='utf-8') as f:
    for i,line in enumerate(f):
        try:
            obj=json.loads(line)
            if 'fusion_label' in obj:
                c[int(obj['fusion_label'])]+=1
                found+=1
        except:
            pass
print('labels found:', found)
print('counts:', dict(c))
