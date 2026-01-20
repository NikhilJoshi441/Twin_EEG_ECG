import requests, socketio, time, json, os

SERVER='http://localhost:5000'
OUT_DIR='data'
OUT_FILE=os.path.join(OUT_DIR,'fusion_raw.jsonl')
MAX=200

os.makedirs(OUT_DIR, exist_ok=True)

print('Fetching alerts list...')
r = requests.get(SERVER + '/api/alerts')
alerts = []
try:
    alerts = r.json().get('alerts', [])
except Exception as e:
    print('Failed to parse /api/alerts', e); raise

print('Found', len(alerts), 'alerts; will collect up to', MAX)

timestamps = [a.get('timestamp') for a in alerts if a.get('timestamp')][:MAX]

sio = socketio.Client()
collected = 0

@sio.event
def connect():
    print('socket connected')

@sio.on('alert_detail')
def on_alert_detail(data):
    global collected
    ts = data.get('ts')
    record = {'ts': ts, 'metrics': data.get('metrics'), 'explanation': data.get('explanation'), 'ecg': data.get('ecg'), 'eeg': data.get('eeg'), 'meta': data.get('meta')}
    with open(OUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    collected += 1
    print('Collected', collected, '->', ts)

@sio.event
def disconnect():
    print('socket disconnected')

print('Connecting socket...')
sio.connect(SERVER)

for ts in timestamps:
    if collected >= MAX:
        break
    try:
        print('Requesting detail for', ts)
        sio.emit('request_alert_detail', {'ts': ts, 'window_s': 2.0})
        # wait for response
        t0 = time.time()
        while collected < MAX and time.time() - t0 < 5:
            time.sleep(0.1)
            # on_alert_detail increments collected
        time.sleep(0.05)
    except Exception as e:
        print('Error requesting', ts, e)

print('Done; collected', collected)
sio.disconnect()
