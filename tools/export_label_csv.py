"""Export label candidate CSV for manual review with small previews and summary stats."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=200)
    args = parser.parse_args()

    Path('data').mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(load_jsonl(args.infile)):
        if i >= args.n:
            break
        ecg = np.asarray(r['ecg'], dtype=float)
        eeg = np.asarray(r['eeg'], dtype=float)
        row = {
            'idx': i,
            'label': r.get('label', ''),
            'ecg_mean': float(np.mean(ecg)),
            'ecg_std': float(np.std(ecg)),
            'eeg_mean': float(np.mean(eeg)),
            'eeg_std': float(np.std(eeg)),
            'ecg_preview': '|'.join([f"{x:.3f}" for x in ecg[:10]]),
            'eeg_preview': '|'.join([f"{x:.3f}" for x in eeg[:10]])
        }
        rows.append(row)

    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['idx'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote', args.out, 'rows=', len(rows))


if __name__ == '__main__':
    main()
