"""Generate synthetic negative/background windows to balance the fusion dataset.

Reads a JSONL file with records containing `ecg`, `eeg`, and `label` fields
and writes an augmented JSONL with additional negative samples created by
noise injection and time-shifting.

Usage: python tools/generate_negatives.py --in data/fusion_raw.jsonl --out data/fusion_augmented.jsonl --target-ratio 1.0
"""
import argparse
import json
import random
import numpy as np
from pathlib import Path


def make_negative(ecg, eeg):
    ecg = np.asarray(ecg, dtype=float).copy()
    eeg = np.asarray(eeg, dtype=float).copy()
    # time-shift a random amount
    for arr in (ecg, eeg):
        if arr.size > 1:
            shift = random.randint(1, max(1, arr.size // 10))
            arr[:] = np.roll(arr, shift)
    # add light gaussian noise
    ecg += np.random.normal(0, 0.01 * (np.std(ecg) + 1e-6), size=ecg.shape)
    eeg += np.random.normal(0, 0.01 * (np.std(eeg) + 1e-6), size=eeg.shape)
    # small scaling
    if random.random() < 0.5:
        ecg *= random.uniform(0.9, 1.1)
    if random.random() < 0.5:
        eeg *= random.uniform(0.9, 1.1)
    return ecg.tolist(), eeg.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--target-ratio", type=float, default=1.0,
                        help="target negative:positive ratio (e.g. 1.0 => balance)")
    args = parser.parse_args()

    p_in = Path(args.infile)
    p_out = Path(args.outfile)
    records = []
    pos = []
    neg = []
    with p_in.open('r', encoding='utf-8') as fh:
        for line in fh:
            obj = json.loads(line)
            records.append(obj)
            if obj.get('label', 1) == 1:
                pos.append(obj)
            else:
                neg.append(obj)

    n_pos = len(pos)
    n_neg = len(neg)
    target_neg = int(round(n_pos * args.target_ratio))
    need = max(0, target_neg - n_neg)

    out = records.copy()
    created = 0
    random.seed(42)
    for i in range(need):
        # sample a positive and convert to negative
        src = random.choice(pos) if pos else random.choice(records)
        ecg, eeg = make_negative(src['ecg'], src['eeg'])
        rec = {'ecg': ecg, 'eeg': eeg, 'label': 0}
        out.append(rec)
        created += 1

    with p_out.open('w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    print(f"Wrote {p_out} total={len(out)} added_negatives={created}")


if __name__ == '__main__':
    main()
