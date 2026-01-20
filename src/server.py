"""Flask + SocketIO server to stream synthetic ECG/EEG in real-time (demo).

This is a simple demo server that emits synthetic samples at regular intervals.
"""
import eventlet
eventlet.monkey_patch()
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time
import os
import numpy as np

from src.simulator import generate_ecg, generate_eeg
from src import preprocessing, feature_extraction
from src import anomaly
from collections import OrderedDict
import json
from datetime import datetime

# serve static files from src/static
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Model paths (configurable via env)
MODEL_RF_PATH = os.environ.get('RF_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'rf.pkl'))
MODEL_LSTM_PATH = os.environ.get('LSTM_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'lstm.pth'))
MODEL_FUSION_PATH = os.environ.get('FUSION_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'fusion_balanced.pth'))
MODEL_FUSION_TS = os.environ.get('FUSION_TS_PATH', os.path.join(BASE_DIR, 'models', 'fusion_balanced_ts.pt'))

# Try loading models if present
rf_model = anomaly.load_rf_model(MODEL_RF_PATH)
if rf_model is not None:
    logging.info(f"Loaded RF model from {MODEL_RF_PATH}")
else:
    logging.info("No RF model found; using heuristic fallback")

lstm_model = None
try:
    lstm_model = anomaly.load_lstm_model(MODEL_LSTM_PATH)
    if lstm_model is not None:
        logging.info(f"Loaded LSTM model from {MODEL_LSTM_PATH}")
    else:
        logging.info("No LSTM model found; skipping LSTM inference")
except Exception:
    logging.exception('Failed loading LSTM model')

# try loading fusion model (prefer TorchScript if present)
fusion_model = None
fusion_ts = None
try:
    if os.path.exists(MODEL_FUSION_TS):
        fusion_ts = anomaly.load_fusion_ts(MODEL_FUSION_TS)
        if fusion_ts is not None:
            fusion_model = fusion_ts
            logging.info(f"Loaded fusion TorchScript model from {MODEL_FUSION_TS}")
    if fusion_model is None:
        fusion_model = anomaly.load_fusion_model(MODEL_FUSION_PATH)
        if fusion_model is not None:
            logging.info(f"Loaded fusion model from {MODEL_FUSION_PATH}")
        else:
            logging.info('No fusion model found; will use RF/LSTM/heuristic pipeline')
except Exception:
    logging.exception('Failed loading fusion model')

# Threat class names for multi-class models (comma-separated in env)
THREAT_CLASSES = os.environ.get('THREAT_CLASSES')
if THREAT_CLASSES:
    THREAT_CLASSES = [c.strip() for c in THREAT_CLASSES.split(',') if c.strip()]
else:
    # default two-class mapping: index 0=normal, 1=threat
    THREAT_CLASSES = ['normal', 'threat']

# Threat log and in-memory history
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
THREAT_LOG = os.path.join(LOG_DIR, 'threats.log')
threat_history = []  # list of recent alerts (summaries)
# full recent entries by timestamp (bounded)
ALERT_ENTRIES = OrderedDict()
ALERT_ENTRIES_MAX = int(os.environ.get('ALERT_ENTRIES_MAX', 500))

# client-side debug beacons (collected for diagnosing client init failures)
CLIENT_DEBUG_LOGS = []
CLIENT_DEBUG_MAX = int(os.environ.get('CLIENT_DEBUG_MAX', 200))

# Log rotation settings
LOG_MAX_BYTES = int(os.environ.get('THREAT_LOG_MAX_BYTES', 5 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.environ.get('THREAT_LOG_BACKUP_COUNT', 3))


def rotate_threat_log():
    try:
        if not os.path.exists(THREAT_LOG):
            return
        size = os.path.getsize(THREAT_LOG)
        if size < LOG_MAX_BYTES:
            return
        # rotate: shift backups
        for i in range(LOG_BACKUP_COUNT - 1, 0, -1):
            s = f"{THREAT_LOG}.{i}"
            d = f"{THREAT_LOG}.{i+1}"
            if os.path.exists(s):
                os.replace(s, d)
        # move current to .1
        os.replace(THREAT_LOG, f"{THREAT_LOG}.1")
    except Exception:
        logging.exception('Failed rotating threat log')


# Recent signal buffer (store recent chunks with timestamps for detail views)
SIGNAL_BUFFER = []
SIGNAL_BUFFER_MAX = int(os.environ.get('SIGNAL_BUFFER_MAX', 400))  # number of chunks to retain


def push_signal_chunk(ts_iso: str, ecg_chunk, eeg_chunk):
    try:
        SIGNAL_BUFFER.append({'ts': ts_iso, 'ecg': list(ecg_chunk), 'eeg': list(eeg_chunk)})
        if len(SIGNAL_BUFFER) > SIGNAL_BUFFER_MAX:
            SIGNAL_BUFFER.pop(0)
    except Exception:
        logging.exception('Failed to push signal chunk')


def human_readable_explanation(expl: dict) -> str:
    try:
        # map internal feature keys -> human readable phrases
        fmap = {
            'LF_HF': 'LF/HF ratio (sympathovagal balance)',
            'SDNN': 'SDNN (global HRV variability)',
            'RMSSD': 'RMSSD (short-term HRV)',
            'spectral_entropy': 'Spectral entropy (signal complexity)',
            'alpha_power': 'EEG alpha power',
            'beta_power': 'EEG beta power',
            'alpha_beta': 'Alpha/Beta power ratio',
            'avg_hr_bpm': 'Average heart rate (bpm)',
            'n_peaks': 'Beat count in window',
        }
        mode = expl.get('model', 'unknown')
        if mode == 'rf':
            top = expl.get('top_features', [])
            if top:
                parts = []
                for p in top:
                    name = fmap.get(p.get('name'), p.get('name'))
                    parts.append(f"{name} ({p.get('value'):.2f}, importance={p.get('importance'):.2f})")
                return f"RF model flagged this alert. Top contributors: {', '.join(parts)}."
            return "RF model flagged this alert."
        elif mode == 'heuristic':
            contrib = expl.get('contributions', {})
            items = sorted(contrib.items(), key=lambda x: -abs(x[1]))
            parts = []
            for k, v in items[:4]:
                name = fmap.get(k, k)
                parts.append(f"{name}: {v:.2f}")
            return f"Heuristic scoring: {', '.join(parts)}." if parts else "Heuristic-based alert."
        elif mode == 'lstm':
            prob = expl.get('prob')
            return f"LSTM model signaled an anomaly (prob={prob:.2f})." if prob is not None else "LSTM model signaled an anomaly."
        else:
            return str(expl)
    except Exception:
        return str(expl)

# Configurable thresholds (can be updated by frontend)
THRESHOLDS = {'high': 0.7, 'medium': 0.35}


def stream_signals():
    # demo loop that emits a short rolling buffer every 0.25s
    # Keep buffers short (128 samples) for plotting
    logging.info("Stream thread started")
    # server-wide flag that can be toggled by control events
    if not hasattr(stream_signals, 'enabled'):
        stream_signals.enabled = True
    while True:
        if stream_signals.enabled:
            # consistent ISO timestamp for this chunk
            ts = datetime.utcnow().isoformat() + 'Z'
            ecg = generate_ecg(0.5, fs=250)  # 0.5s worth
            eeg = generate_eeg(0.5, fs=128)
            # compute light-weight metrics to emit alongside signals
            try:
                # preprocess
                ecg_filt = preprocessing.bandpass_filter(ecg, low=0.5, high=40.0, fs=250)
                ecg_filt = preprocessing.notch_filter(ecg_filt, fs=250, freq=50.0)
                peaks = feature_extraction.pan_tompkins_detector(ecg_filt, fs=250)
                rr = feature_extraction.compute_rr_intervals(peaks, fs=250)
                hrv_td = feature_extraction.hrv_time_domain(rr)
                hrv_fd = feature_extraction.hrv_frequency_domain(rr)
            except Exception as e:
                logging.exception('Error computing metrics: %s', e)
                peaks = []
                rr = []
                hrv_td = {}
                hrv_fd = {}

            try:
                eeg_filt = preprocessing.bandpass_filter(eeg, low=0.5, high=45.0, fs=128)
                eeg_filt = preprocessing.notch_filter(eeg_filt, fs=128, freq=50.0)
                bands = feature_extraction.compute_bandpowers(eeg_filt, fs=128)
                entropy = feature_extraction.spectral_entropy(eeg_filt, fs=128)
            except Exception:
                logging.exception('Error computing EEG features')
                bands = {}
                entropy = 0.0

            metrics = {
                'n_peaks': int(len(peaks)),
                'avg_hr_bpm': float(feature_extraction.compute_hr_from_rr(rr)) if rr else 0.0,
                'SDNN': float(hrv_td.get('SDNN', 0.0)) if isinstance(hrv_td, dict) else 0.0,
                'RMSSD': float(hrv_td.get('RMSSD', 0.0)) if isinstance(hrv_td, dict) else 0.0,
                'pNN50': float(hrv_td.get('pNN50', 0.0)) if isinstance(hrv_td, dict) else 0.0,
                'LF_HF': float(hrv_fd.get('LF_HF', 0.0)) if isinstance(hrv_fd, dict) else 0.0,
                'alpha_power': float(bands.get('alpha', 0.0)) if isinstance(bands, dict) else 0.0,
                'beta_power': float(bands.get('beta', 0.0)) if isinstance(bands, dict) else 0.0,
                'spectral_entropy': float(entropy),
            }

            # Build feature vector (consistent order)
            def clamp(v, a=0.0, b=1.0):
                try:
                    return max(a, min(b, float(v)))
                except Exception:
                    return a

            lf_hf = metrics.get('LF_HF', 0.0)
            sdnn = metrics.get('SDNN', 0.0)
            rmssd = metrics.get('RMSSD', 0.0)
            entropy_val = metrics.get('spectral_entropy', 0.0)
            alpha = metrics.get('alpha_power', 0.0)
            beta = metrics.get('beta_power', 0.0)
            alpha_beta = (alpha / (beta + 1e-12)) if beta >= 0 else 0.0

            feat_names = ['LF_HF', 'SDNN', 'RMSSD', 'spectral_entropy', 'alpha_power', 'beta_power', 'alpha_beta', 'avg_hr_bpm', 'n_peaks']
            vals = [lf_hf, sdnn, rmssd, entropy_val, alpha, beta, alpha_beta, metrics.get('avg_hr_bpm', 0.0), metrics.get('n_peaks', 0)]
            X = np.array(vals, dtype=float).reshape(1, -1)

            explanation = {}
            threat_score = None

            # Try RF model first
            if rf_model is not None:
                try:
                    prob = anomaly.predict_rf(rf_model, X)[0]
                    threat_score = float(prob)
                    explanation['model'] = 'rf'
                    explanation['prob'] = threat_score
                    imps = anomaly.get_rf_feature_importances(rf_model, feat_names)
                    top = sorted(imps.items(), key=lambda x: -x[1])[:3]
                    explanation['top_features'] = [{'name': n, 'importance': float(imps.get(n, 0.0)), 'value': float(vals[feat_names.index(n)])} for n, _ in top]
                except Exception:
                    logging.exception('RF model inference failed; falling back')

            # If no RF result and LSTM available, try LSTM
            if threat_score is None and lstm_model is not None:
                try:
                    Xl = X.reshape(1, 1, -1)
                    prob = anomaly.predict_lstm(lstm_model, Xl)[0]
                    threat_score = float(prob)
                    explanation['model'] = 'lstm'
                    explanation['prob'] = threat_score
                except Exception:
                    logging.exception('LSTM inference failed; falling back')

            # If still None, try fusion model if available (requires raw ecg/eeg windows)
            if threat_score is None and fusion_model is not None:
                try:
                    # prepare simple fixed-length windows (pad/truncate)
                    ecg_arr = np.array(ecg, dtype=float)
                    eeg_arr = np.array(eeg, dtype=float)
                    # naive windowing: take first ECG_LEN/EEG_LEN samples or pad with zeros
                    ECG_LEN = 500
                    EEG_LEN = 256
                    ecg_w = ecg_arr[:ECG_LEN].tolist() + [0.0] * max(0, ECG_LEN - len(ecg_arr))
                    eeg_w = eeg_arr[:EEG_LEN].tolist() + [0.0] * max(0, EEG_LEN - len(eeg_arr))
                    pred = anomaly.predict_fusion(fusion_model, np.array([ecg_w]), np.array([eeg_w]))
                    # pred can be:
                    # - 1D array of shape (N,) -> binary probability per sample
                    # - 2D array of shape (N, C) -> multiclass probabilities per sample
                    try:
                        pred_arr = np.asarray(pred)
                        if pred_arr.ndim == 1:
                            threat_score = float(pred_arr[0])
                            explanation['model'] = 'fusion'
                            explanation['prob'] = threat_score
                        elif pred_arr.ndim == 2:
                            probs = pred_arr[0].tolist()
                            # pick predicted class
                            idx = int(np.argmax(probs))
                            cls_name = THREAT_CLASSES[idx] if idx < len(THREAT_CLASSES) else f'class_{idx}'
                            # for binary models, keep threat_score as prob of index 1 (threat)
                            if len(probs) == 2:
                                threat_score = float(probs[1])
                            else:
                                threat_score = float(max(probs))
                            explanation['model'] = 'fusion'
                            explanation['predicted_class'] = cls_name
                            explanation['class_probs'] = {THREAT_CLASSES[i] if i < len(THREAT_CLASSES) else f'class_{i}': float(p) for i, p in enumerate(probs)}
                            explanation['prob'] = threat_score
                        else:
                            threat_score = float(np.asarray(pred).reshape(-1)[0])
                            explanation['model'] = 'fusion'
                            explanation['prob'] = threat_score
                    except Exception:
                        # fallback to previous single-value behavior
                        try:
                            prob = np.asarray(pred).reshape(-1)[0]
                            threat_score = float(prob)
                            explanation['model'] = 'fusion'
                            explanation['prob'] = threat_score
                        except Exception:
                            raise
                except Exception:
                    logging.exception('Fusion model inference failed; falling back')

            # fallback to heuristic
            if threat_score is None:
                contrib = {}
                contrib['lf_hf'] = clamp((lf_hf - 1.0) / 4.0) * 0.25
                contrib['sdnn'] = clamp((0.08 - sdnn) / 0.08) * 0.2
                contrib['rmssd'] = clamp((0.03 - rmssd) / 0.03) * 0.2
                contrib['entropy'] = clamp((entropy_val - 3.0) / 3.0) * 0.15
                contrib['alpha_beta'] = clamp((alpha_beta - 1.0) / 2.0) * 0.2
                s = sum(contrib.values())
                threat_score = max(0.0, min(1.0, s))
                explanation['model'] = 'heuristic'
                explanation['contributions'] = contrib

            # Determine level using configurable thresholds
            if threat_score >= THRESHOLDS.get('high', 0.7):
                threat_level = 'high'
                color = 'red'
            elif threat_score >= THRESHOLDS.get('medium', 0.35):
                threat_level = 'medium'
                color = 'orange'
            else:
                threat_level = 'low'
                color = 'green'

            threat = {
                'score': float(threat_score),
                'level': threat_level,
                'color': color,
                'model': explanation.get('model', 'none'),
            }
            # expose predicted class if present in explanation (useful for UI)
            if 'predicted_class' in explanation:
                threat['predicted_class'] = explanation.get('predicted_class')

            # persist alert to log (use same timestamp as chunk)
            try:
                # rotate log if needed before writing
                rotate_threat_log()
                # pack some metadata (sampling info, last R-peak index in this chunk)
                r_peak_idx = int(peaks[-1]) if isinstance(peaks, (list, tuple, np.ndarray)) and len(peaks) > 0 else None
                entry = {
                    'timestamp': ts,
                    'metrics': metrics,
                    'threat': threat,
                    'explanation': explanation,
                    'meta': {
                        'ecg_fs': 250,
                        'eeg_fs': 128,
                        'r_peak_index_in_chunk': r_peak_idx,
                        'origin_chunk_ts': ts,
                    }
                }
                # write to file
                with open(THREAT_LOG, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
                # keep in-memory mapping for quick lookup
                try:
                    ALERT_ENTRIES[entry['timestamp']] = entry
                    # maintain bounded size
                    while len(ALERT_ENTRIES) > ALERT_ENTRIES_MAX:
                        ALERT_ENTRIES.popitem(last=False)
                except Exception:
                    logging.exception('Failed to insert into ALERT_ENTRIES')
            except Exception:
                logging.exception('Failed to write threat log')

            # update in-memory history (keep recent 200)
            try:
                short = {'ts': entry['timestamp'], 'score': threat['score'], 'level': threat['level'], 'model': threat.get('model'), 'explanation': explanation}
                threat_history.append(short)
                if len(threat_history) > 200:
                    threat_history.pop(0)
            except Exception:
                pass

            # attach the same timestamped chunk to buffer
            push_signal_chunk(ts, ecg, eeg)
            # emit signal, metrics, threat and history
            socketio.emit("signal", {"ecg": ecg, "eeg": eeg, 'ts': ts})
            socketio.emit("metrics", metrics)
            socketio.emit("threat", threat)
            socketio.emit('threat_history', threat_history[-50:])
            socketio.sleep(0.25)  # type: ignore


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route('/alerts')
def alerts_page():
    # serve the alerts management page with an initial embedded alerts payload
    try:
        out = []
        # prefer recent in-memory entries first
        for ts, ent in reversed(list(ALERT_ENTRIES.items())):
            out.append(ent)
        # include persisted file entries (avoid duplicates)
        if os.path.exists(THREAT_LOG):
            with open(THREAT_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get('timestamp') not in ALERT_ENTRIES:
                            out.append(obj)
                    except Exception:
                        continue
        # limit to most recent 200
        initial = out[:200]
    except Exception:
        logging.exception('Failed preparing initial alerts for page')
        initial = []
    # attempt to read the client script so we can inline it into the page
    alerts_js_text = None
    try:
        alerts_js_path = os.path.join(STATIC_DIR, 'js', 'alerts.js')
        if os.path.exists(alerts_js_path):
            with open(alerts_js_path, 'r', encoding='utf-8') as f:
                alerts_js_text = f.read()
    except Exception:
        logging.exception('Failed reading alerts.js for inline embedding')
    return render_template('alerts.html', initial_alerts=initial, alerts_js=alerts_js_text)


@app.route('/api/alerts')
def api_alerts():
    # return JSON list of alerts; optional filters via query params: level, model, since, until
    try:
        level = None
        model = None
        since = None
        until = None
        from flask import request
        level = request.args.get('level')
        model = request.args.get('model')
        since = request.args.get('since')
        until = request.args.get('until')
        out = []
        # prefer reading latest in-memory entries first
        for ts, ent in reversed(list(ALERT_ENTRIES.items())):
            out.append(ent)
        # fallback: include persisted file if requested
        if os.path.exists(THREAT_LOG):
            with open(THREAT_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        # avoid duplicates (in-memory wins)
                        if obj.get('timestamp') not in ALERT_ENTRIES:
                            out.append(obj)
                    except Exception:
                        continue
        # apply simple filters
        def keep(e):
            if level and (e.get('threat', {}).get('level') != level):
                return False
            if model and (e.get('threat', {}).get('model') != model and e.get('explanation', {}).get('model') != model):
                return False
            # time filters (ISO)
            try:
                if since and e.get('timestamp') < since:
                    return False
                if until and e.get('timestamp') > until:
                    return False
            except Exception:
                pass
            return True

        filtered = [e for e in out if keep(e)]
        return jsonify({'count': len(filtered), 'alerts': filtered})
    except Exception:
        logging.exception('Failed to list alerts')
        return jsonify({'error': 'failed'}), 500


@app.route('/api/client_debug', methods=['POST'])
def api_client_debug():
    try:
        from flask import request
        obj = request.get_json(force=True, silent=True)
        if obj is None:
            obj = {'raw': request.get_data(as_text=True)}
        obj['_ts'] = datetime.utcnow().isoformat() + 'Z'
        CLIENT_DEBUG_LOGS.append(obj)
        # bound size
        while len(CLIENT_DEBUG_LOGS) > CLIENT_DEBUG_MAX:
            CLIENT_DEBUG_LOGS.pop(0)
        logging.info('Received client debug: %s', obj.get('event') or obj.get('raw') or '')
        return jsonify({'ok': True})
    except Exception:
        logging.exception('Failed receiving client debug')
        return jsonify({'error': 'failed'}), 500


@app.route('/api/client_debug/logs')
def api_client_debug_logs():
    try:
        return jsonify({'count': len(CLIENT_DEBUG_LOGS), 'logs': list(CLIENT_DEBUG_LOGS)})
    except Exception:
        logging.exception('Failed returning client debug logs')
        return jsonify({'error': 'failed'}), 500


@app.route('/api/alerts/export')
def api_alerts_export():
    try:
        import csv
        from flask import Response, request
        level = request.args.get('level')
        model = request.args.get('model')
        since = request.args.get('since')
        until = request.args.get('until')
        out = []
        for ts, ent in reversed(list(ALERT_ENTRIES.items())):
            out.append(ent)
        if os.path.exists(THREAT_LOG):
            with open(THREAT_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get('timestamp') not in ALERT_ENTRIES:
                            out.append(obj)
                    except Exception:
                        continue

        def keep(e):
            if level and (e.get('threat', {}).get('level') != level):
                return False
            if model and (e.get('threat', {}).get('model') != model and e.get('explanation', {}).get('model') != model):
                return False
            try:
                if since and e.get('timestamp') < since:
                    return False
                if until and e.get('timestamp') > until:
                    return False
            except Exception:
                pass
            return True

        alerts = [e for e in out if keep(e)]
        # build CSV
        def gen():
            w = None
            for a in alerts:
                flat = {
                    'timestamp': a.get('timestamp'),
                    'level': a.get('threat', {}).get('level'),
                    'score': a.get('threat', {}).get('score'),
                    'model': a.get('threat', {}).get('model'),
                }
                # include metrics flattened as JSON
                flat['metrics'] = json.dumps(a.get('metrics', {}))
                if w is None:
                    header = list(flat.keys())
                    yield ','.join(header) + '\n'
                    w = header
                yield ','.join('"%s"' % str(flat.get(k, '')) for k in w) + '\n'

        return Response(gen(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename="alerts.csv"'})
    except Exception:
        logging.exception('Failed to export alerts')
        return jsonify({'error': 'failed'}), 500


@socketio.on("connect")
def on_connect():
    logging.info("Client connected")
    emit("connected", {"message": "connected"})
    # send recent history and thresholds on connect
    try:
        emit('threat_history', threat_history[-50:])
        emit('thresholds', THRESHOLDS)
    except Exception:
        pass


@socketio.on('control')
def on_control(data):
    """Handle control messages from frontend: {action: 'start'|'stop'}"""
    action = data.get('action') if isinstance(data, dict) else None
    if action == 'stop':
        stream_signals.enabled = False
        logging.info('Streaming stopped by client')
    elif action == 'start':
        stream_signals.enabled = True
        logging.info('Streaming started by client')
    # allow updating thresholds: {thresholds: {high:0.7, medium:0.35}}
    if isinstance(data, dict) and 'thresholds' in data:
        t = data.get('thresholds') or {}
        try:
            if isinstance(t, dict):
                if 'high' in t and t['high'] is not None:
                    THRESHOLDS['high'] = float(t['high'])
                if 'medium' in t and t['medium'] is not None:
                    THRESHOLDS['medium'] = float(t['medium'])
                logging.info('Updated thresholds: %s', THRESHOLDS)
                emit('thresholds', THRESHOLDS)
        except Exception:
            logging.exception('Failed to update thresholds')


@socketio.on('request_alert_detail')
def on_request_alert_detail(data):
    """Client requests detailed alert info including nearby raw signals.

    Expects: {'ts': <timestamp ISO>} and optional {'window_s': <seconds>}.
    Responds with 'alert_detail' event containing: {ts, metrics, explanation, ecg: [...], eeg: [...]}.
    """
    ts = None
    try:
        if isinstance(data, dict):
            ts = data.get('ts')
            window_s = float(data.get('window_s', 2.0))
        else:
            window_s = 2.0
    except Exception:
        ts = None
        window_s = 2.0

    if ts is None:
        emit('alert_detail', {'error': 'missing ts'})
        return

    # find full entry in in-memory map first
    entry = ALERT_ENTRIES.get(ts)
    if entry is None:
        # try reading from threat log (last lines)
        try:
            if os.path.exists(THREAT_LOG):
                with open(THREAT_LOG, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if obj.get('timestamp') == ts:
                                entry = obj
                                break
                        except Exception:
                            continue
        except Exception:
            logging.exception('Failed reading threat log for detail')

    if entry is None:
        emit('alert_detail', {'error': 'alert not found'})
        return

    # parse ts and build window center
    try:
        from dateutil import parser as _p
        center = _p.isoparse(ts)
    except Exception:
        try:
            center = datetime.fromisoformat(ts.rstrip('Z'))
        except Exception:
            center = None

    ecg_agg = []
    eeg_agg = []
    r_peak_pos = None
    if center is not None:
        # iterate SIGNAL_BUFFER in chronological order
        for chunk in SIGNAL_BUFFER:
            try:
                t = datetime.fromisoformat(chunk['ts'].rstrip('Z'))
                delta = (t - center).total_seconds()
                if abs(delta) <= window_s:
                    # if this chunk matches the origin_chunk_ts for the alert, mark r-peak
                    if entry.get('meta', {}).get('origin_chunk_ts') == chunk.get('ts'):
                        # r_peak index within this chunk
                        r_idx = entry.get('meta', {}).get('r_peak_index_in_chunk')
                        if r_idx is not None:
                            # position in aggregated array will be current length + r_idx
                            r_peak_pos = len(ecg_agg) + int(r_idx)
                    ecg_agg.extend(chunk.get('ecg', []))
                    eeg_agg.extend(chunk.get('eeg', []))
            except Exception:
                continue

    detail = {
        'ts': ts,
        'metrics': entry.get('metrics'),
        'explanation': entry.get('explanation'),
        'explanation_text': human_readable_explanation(entry.get('explanation') or {}),
        'ecg': ecg_agg,
        'eeg': eeg_agg,
        'r_peak_pos': r_peak_pos,
        'meta': entry.get('meta', {}),
    }
    emit('alert_detail', detail)


def start_stream_thread():
    # Use SocketIO's background task which integrates with the chosen async mode
    socketio.start_background_task(stream_signals)
    logging.info("Requested background streaming task")


if __name__ == "__main__":
    start_stream_thread()
    socketio.run(app, host="0.0.0.0", port=5000)
