import json
from pathlib import Path

def main():
    p = Path('data/fusion_augmented.jsonl')
    counts = {}
    total = 0
    with p.open('r', encoding='utf-8') as fh:
        for line in fh:
            total += 1
            obj = json.loads(line)
            lab = obj.get('label', 1)
            counts[lab] = counts.get(lab, 0) + 1
    print('total=', total, 'counts=', counts)

if __name__ == '__main__':
    main()
