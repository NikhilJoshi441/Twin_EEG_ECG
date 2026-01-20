"""Run a small hyperparameter sweep by invoking train_kfold.py with different params.

Usage: python tools/hyperparam_sweep.py --data data/fusion_augmented_v2.jsonl
"""
import argparse
import subprocess
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='reports/hyperparam_summary.json')
    args = parser.parse_args()
    combos = [
        {'folds':5,'epochs':10,'batch':32},
        {'folds':5,'epochs':20,'batch':32},
        {'folds':5,'epochs':20,'batch':64}
    ]
    results = []
    for c in combos:
        cmd = ["python","tools/train_kfold.py","--data",args.data,"--out","reports/kfold_tmp.json","--folds",str(c['folds']),"--epochs",str(c['epochs']),"--batch",str(c['batch'])]
        print('Running',cmd)
        subprocess.run(cmd, check=True)
        with open('reports/kfold_tmp.json','r',encoding='utf-8') as fh:
            r = json.load(fh)
        results.append({'params':c,'report':r})
    Path('reports').mkdir(exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print('Saved', args.out)

if __name__ == '__main__':
    main()
