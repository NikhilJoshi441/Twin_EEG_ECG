"""Retrain RandomForest from labeled features or from threat logs.

Usage:
  python tools/retrain_rf.py --features features.csv --label label_col --out src/models/rf.pkl
  or
  python tools/retrain_rf.py --from-logs --out src/models/rf.pkl

If no features file provided, the script will try to assemble a dataset from
`src/logs/threats.log` using the metrics field and label examples using the
existing threat score (thresholded) as noisy labels. The script will evaluate
on a held-out split and print metrics.
"""
import argparse
import os
import json
import numpy as np

# Ensure project root is on sys.path so `import src` works when running this script
import sys
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src import anomaly

def load_from_csv(path, label_col='label'):
    import pandas as pd
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError('label_col not in CSV')
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).values.astype(float)
    return X, y

def load_from_logs(log_path, score_threshold=0.5):
    X = []
    y = []
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                metrics = obj.get('metrics') or {}
                # flatten chosen features; fallback to common names
                feat_names = ['LF_HF', 'SDNN', 'RMSSD', 'spectral_entropy', 'alpha_power', 'beta_power', 'alpha_beta', 'avg_hr_bpm', 'n_peaks']
                row = [float(metrics.get(fn, 0.0)) for fn in feat_names]
                X.append(row)
                score = float(obj.get('threat', {}).get('score', 0.0))
                label = 1 if score >= score_threshold else 0
                y.append(label)
            except Exception:
                continue
    return np.array(X, dtype=float), np.array(y, dtype=int)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', help='CSV of features with label column')
    p.add_argument('--label', default='label', help='Label column name for CSV')
    p.add_argument('--from-logs', action='store_true', help='Assemble dataset from threat logs')
    p.add_argument('--out', default='src/models/rf.pkl', help='Output model path')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--score-threshold', type=float, default=0.6)
    args = p.parse_args()

    X = None; y = None
    if args.features:
        X, y = load_from_csv(args.features, label_col=args.label)
    elif args.from_logs:
        log_path = os.path.join('src', 'logs', 'threats.log')
        X, y = load_from_logs(log_path, score_threshold=args.score_threshold)
    else:
        p.print_help(); return

    if len(X) == 0:
        print('No data loaded'); return

    # train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y if len(set(y))>1 else None)

    print(f'Training on {len(X_train)} samples, evaluating on {len(X_test)} samples')
    model = anomaly.train_rf_baseline(X_train, y_train, n_estimators=200, random_state=args.seed)
    try:
        from sklearn.metrics import classification_report, roc_auc_score
        proba = None
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            # fallback to decision_function or predict
            try:
                df = model.decision_function(X_test)
                # convert to probability-like via sigmoid
                import numpy as _np
                probs = 1.0 / (1.0 + _np.exp(-_np.array(df)))
            except Exception:
                probs = model.predict(X_test).astype(float)

        if proba is not None:
            # handle classifiers with single-prob column
            if proba.shape[1] > 1:
                probs = proba[:, 1]
            else:
                probs = proba[:, 0]

        preds = (probs >= 0.5).astype(int)
        print(classification_report(y_test, preds))
        if len(set(y_test)) > 1:
            try:
                print('ROC AUC:', roc_auc_score(y_test, probs))
            except Exception:
                pass
    except Exception as e:
        import traceback
        print('Failed to compute metrics; exception follows:')
        traceback.print_exc()

    anomaly.save_rf_model(model, args.out)
    print('Saved model to', args.out)

if __name__ == '__main__':
    main()
