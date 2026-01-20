"""Stronger augmentation pipeline for generating synthetic negative/background windows.

Adds time-shifts, noise, scaling, random cropping and mixing of signals.
"""
import argparse
import json
import random
import numpy as np
from pathlib import Path


def augment_negative(ecg, eeg, noise_scale=0.02, mix_prob=0.5, invert_prob=0.1, max_shift_frac=0.1):
    ecg = np.asarray(ecg, dtype=float).copy()
    eeg = np.asarray(eeg, dtype=float).copy()
    L_ecg = len(ecg)
    L_eeg = len(eeg)
    # random crop and pad
    def crop_pad(x, target):
        if len(x) > target:
            start = random.randint(0, len(x) - target)
            return x[start:start+target]
        elif len(x) < target:
            pad = np.zeros(target - len(x))
            return np.concatenate([x, pad])
        return x

    ecg = crop_pad(ecg, L_ecg)
    eeg = crop_pad(eeg, L_eeg)

    # time warp: small roll
    max_shift_ecg = max(1, int(L_ecg * max_shift_frac))
    max_shift_eeg = max(1, int(L_eeg * max_shift_frac))
    shift_ecg = random.randint(-max_shift_ecg, max_shift_ecg)
    shift_eeg = random.randint(-max_shift_eeg, max_shift_eeg)
    ecg = np.roll(ecg, shift_ecg)
    eeg = np.roll(eeg, shift_eeg)

    # scaling & inversion
    ecg *= random.uniform(0.8, 1.2)
    eeg *= random.uniform(0.8, 1.2)
    if random.random() < invert_prob:
        ecg = -ecg
    if random.random() < invert_prob:
        eeg = -eeg

    # add gaussian noise
    ecg += np.random.normal(0, noise_scale * (np.std(ecg) + 1e-6), size=ecg.shape)
    eeg += np.random.normal(0, noise_scale * (np.std(eeg) + 1e-6), size=eeg.shape)

    # mix with another random signal sometimes
    if random.random() < mix_prob and len(all_pos) > 0:
        mix_factor = random.uniform(0.05, 0.35)
        idx = random.randint(0, len(all_pos) - 1)
        other = np.asarray(all_pos[idx]['ecg'], dtype=float)
        other = np.resize(other, L_ecg)
        ecg = (1 - mix_factor) * ecg + mix_factor * other

    return ecg.tolist(), eeg.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    parser.add_argument('--target-ratio', type=float, default=1.0)
    parser.add_argument('--noise-scale', type=float, default=0.02)
    parser.add_argument('--mix-prob', type=float, default=0.5)
    parser.add_argument('--invert-prob', type=float, default=0.1)
    parser.add_argument('--max-shift-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
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

    global all_pos
    all_pos = pos if pos else records

    n_pos = len(pos)
    n_neg = len(neg)
    target_neg = int(round(n_pos * args.target_ratio))
    need = max(0, target_neg - n_neg)

    out = records.copy()
    created = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    for i in range(need):
        src = random.choice(all_pos)
        ecg, eeg = augment_negative(src['ecg'], src['eeg'], noise_scale=args.noise_scale, mix_prob=args.mix_prob, invert_prob=args.invert_prob, max_shift_frac=args.max_shift_frac)
        rec = {'ecg': ecg, 'eeg': eeg, 'label': 0}
        out.append(rec)
        created += 1

    with p_out.open('w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    print(f'Wrote {p_out} total={len(out)} added_negatives={created}')


if __name__ == '__main__':
    main()
