"""Plot sample ECG/EEG windows for quick inspection and save PNG.

Usage: python tools/visualize_samples.py --in data/fusion_augmented.jsonl --out reports/samples.png --n 6
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=6)
    args = parser.parse_args()

    recs = load_jsonl(args.infile)
    Path('reports').mkdir(exist_ok=True)
    n = min(args.n, len(recs))
    fig, axs = plt.subplots(n, 2, figsize=(10, 3 * n))
    for i in range(n):
        r = recs[i]
        ecg = np.asarray(r['ecg'], dtype=float)
        eeg = np.asarray(r['eeg'], dtype=float)
        lab = r.get('label', None)
        axs[i, 0].plot(ecg)
        axs[i, 0].set_title(f'ECG idx={i} label={lab}')
        axs[i, 1].plot(eeg)
        axs[i, 1].set_title(f'EEG idx={i} label={lab}')
    plt.tight_layout()
    plt.savefig(args.out)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
