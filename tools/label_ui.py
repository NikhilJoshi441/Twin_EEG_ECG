"""Upgraded Flask labeling UI for sample inspection and manual labeling.

Run: python tools/label_ui.py
Visit: http://localhost:5001
"""
from flask import Flask, jsonify, request, send_file, render_template_string
import csv
import json
from pathlib import Path

app = Flask(__name__)
DATA_CSV = Path('data/label_candidates.csv')
JSONL_CANDS = [
    Path('data/fusion_augmented_v3.jsonl'),
    Path('data/fusion_augmented_v2.jsonl'),
    Path('data/fusion_augmented.jsonl'),
    Path('data/fusion_raw.jsonl')
]
DATA_JSONL = None
for p in JSONL_CANDS:
    if p.exists():
        DATA_JSONL = p
        break

MANUAL_OUT = Path('data/manual_labels.csv')


def load_rows():
    rows = []
    if DATA_CSV.exists():
        with open(DATA_CSV, 'r', encoding='utf-8') as fh:
            r = csv.DictReader(fh)
            for row in r:
                rows.append(row)
    return rows


def load_jsonl_records():
    recs = []
    if DATA_JSONL is None:
        return recs
    with open(DATA_JSONL, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                recs.append({})
    return recs


def load_manual_labels():
    labels = {}
    if MANUAL_OUT.exists():
        with open(MANUAL_OUT, 'r', encoding='utf-8') as fh:
            r = csv.DictReader(fh)
            for row in r:
                try:
                    labels[int(row['idx'])] = int(row['label'])
                except Exception:
                    continue
    return labels


JSONL_RECORDS = load_jsonl_records()


@app.route('/stats')
def stats():
    rows = load_rows()
    total = len(rows)
    manual = load_manual_labels()
    labeled = len(manual)
    next_unlabeled = -1
    for i in range(total):
        if i not in manual:
            next_unlabeled = i
            break
    return jsonify({'total': total, 'labeled': labeled, 'next_unlabeled': next_unlabeled})


@app.route('/next_unlabeled')
def next_unlabeled():
    s = stats().get_json()
    return jsonify({'next': s.get('next_unlabeled', -1)})


@app.route('/download_labels')
def download_labels():
    if not MANUAL_OUT.exists():
        return ('', 404)
    return send_file(str(MANUAL_OUT), as_attachment=True)


@app.route('/')
def index():
    return render_template_string('''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Labeling UI</title>
  <style>body{font-family:Arial,Helvetica,sans-serif;margin:16px} .btn{margin:4px;padding:6px 10px}</style>
</head>
<body>
<h2>Labeling UI</h2>
<div><span id="stats">Loading...</span> &nbsp; <a href="/download_labels">Download labels</a></div>
<div style="margin-top:12px">
  <button class="btn" onclick="prev()">Prev</button>
  <button class="btn" onclick="next()">Next</button>
  <button class="btn" onclick="jumpTo()">Jump</button>
  <input id="jumpIdx" style="width:80px" placeholder="index" />
  <button class="btn" onclick="gotoNextUnlabeled()">Next Unlabeled</button>
</div>
<div id=app style="margin-top:12px"></div>
<div style="margin-top:12px">
  <button class="btn" onclick="save(1)">Positive</button>
  <button class="btn" onclick="save(0)">Negative</button>
  <button class="btn" onclick="save(2)">Uncertain</button>
  <button class="btn" onclick="skip()">Skip</button>
</div>

<script>
let idx=0;
async function refreshStats(){
  const r = await fetch('/stats'); const s = await r.json();
  document.getElementById('stats').innerText = `Total: ${s.total}  Labeled: ${s.labeled}  Next unlabeled: ${s.next_unlabeled}`;
  return s;
}

async function load(i){
  const r=await fetch('/sample/'+i);
  if(!r.ok){document.getElementById('app').innerText='No more'; return}
  const j=await r.json();
  idx = i;
  let html = `<h3>Index ${i}  (label: ${j.label || 'none'})</h3>`;
  if(j.ecg_preview || j.eeg_preview){
    html += `<div><strong>ECG:</strong><pre style="max-height:120px;overflow:auto">${j.ecg_preview}</pre></div>`;
    html += `<div><strong>EEG:</strong><pre style="max-height:120px;overflow:auto">${j.eeg_preview}</pre>`;
  } else if (j.metrics){
    html += `<pre>metrics:\n${JSON.stringify(j.metrics,null,2)}\nexplanation:\n${JSON.stringify(j.explanation||{},null,2)}</pre>`;
  } else {
    html += `<div>No preview available</div>`;
  }
  document.getElementById('app').innerHTML = html;
  await refreshStats();
}

async function save(v){
  await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx:idx,label:v})});
  await refreshStats();
  next();
}

function skip(){ next(); }

async function prev(){ if(idx>0) load(idx-1); }
async function next(){ load(idx+1); }

function jumpTo(){ const v=document.getElementById('jumpIdx').value; const n=parseInt(v); if(!isNaN(n)) load(n); }

async function gotoNextUnlabeled(){ const r=await fetch('/next_unlabeled'); const j=await r.json(); if(j.next>=0) load(j.next); }

refreshStats().then(s=>{ if(s.next_unlabeled>=0) load(s.next_unlabeled); else load(0); });
</script>
</body>
</html>
''')


@app.route('/sample/<int:i>')
def sample(i):
    rows = load_rows()
    if i<0 or i>=len(rows):
        return ('',404)
    row = rows[i]
    if JSONL_RECORDS and i < len(JSONL_RECORDS):
        rec = JSONL_RECORDS[i]
        ecg = rec.get('ecg', [])
        eeg = rec.get('eeg', [])
        ecg_preview = '|'.join([f"{x:.3f}" for x in ecg[:60]]) if ecg else ''
        eeg_preview = '|'.join([f"{x:.3f}" for x in eeg[:60]]) if eeg else ''
        row.setdefault('ecg_preview', ecg_preview)
        row.setdefault('eeg_preview', eeg_preview)
        row['metrics'] = rec.get('metrics', {})
        row['explanation'] = rec.get('explanation', {})
    return jsonify(row)


@app.route('/label', methods=['POST'])
def label():
    j = request.get_json()
    MANUAL_OUT.parent.mkdir(exist_ok=True)
    write_header = not MANUAL_OUT.exists()
    with open(MANUAL_OUT, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(['idx','label'])
        w.writerow([j.get('idx'), j.get('label')])
    return ('',204)


if __name__ == '__main__':
    app.run(port=5001)
"""Simple Flask app to review and label candidate samples.

Usage: python tools/label_ui.py
Visit http://localhost:5001 in a browser to review and label samples.
Labels are saved to `data/manual_labels.csv`.
"""
from flask import Flask, jsonify, request, send_file, render_template_string
import csv
import json
from pathlib import Path
import io

app = Flask(__name__)
DATA_CSV = Path('data/label_candidates.csv')
# try a few likely JSONL sources to fetch full samples for preview
JSONL_CANDS = [
    Path('data/fusion_augmented_v3.jsonl'),
    Path('data/fusion_augmented_v2.jsonl'),
    Path('data/fusion_augmented.jsonl'),
    Path('data/fusion_raw.jsonl')
]
DATA_JSONL = None
for p in JSONL_CANDS:
    if p.exists():
        DATA_JSONL = p
        break
MANUAL_OUT = Path('data/manual_labels.csv')

def load_rows():
    rows = []
    if DATA_CSV.exists():
        import csv
        with open(DATA_CSV, 'r', encoding='utf-8') as fh:
            r = csv.DictReader(fh)
            for row in r:
                rows.append(row)
    return rows


def load_jsonl_records():
    recs = []
    if DATA_JSONL is None:
        return recs
    with open(DATA_JSONL, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                recs.append({})
    return recs


# load JSONL records once
JSONL_RECORDS = load_jsonl_records()

@app.route('/')
def index():
        return render_template_string('''
<!doctype html>
<title>Labeling UI</title>
<div id=app></div>
<script>
let idx=0; async function load(i){
    const r=await fetch('/sample/'+i); if(!r.ok){document.getElementById('app').innerText='No more'; return}
    const j=await r.json();
    let metrics = j.metrics ? JSON.stringify(j.metrics, null, 2) : '';
    let expl = j.explanation ? JSON.stringify(j.explanation, null, 2) : '';
    let preview = '';
    if(j.ecg_preview || j.eeg_preview){
        preview = `<pre>ECG preview: ${j.ecg_preview}\nEEG preview: ${j.eeg_preview}</pre>`;
    } else if (metrics) {
        preview = `<pre>metrics:\n${metrics}\nexplanation:\n${expl}</pre>`;
    }
    document.getElementById('app').innerHTML=`<h3>idx ${i} label ${j.label}</h3>${preview}<button onclick="save(1)">Positive</button> <button onclick="save(0)">Negative</button> <button onclick="next()">Skip</button>`;
}
async function save(v){ await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx:idx,label:v})}); next(); }
function next(){ idx++; load(idx); }
load(0);
</script>
''')

@app.route('/sample/<int:i>')
def sample(i):
    rows = load_rows()
    if i<0 or i>=len(rows):
        return ('',404)
    row = rows[i]
    # if previews missing, try to fetch from JSONL_RECORDS
    if JSONL_RECORDS and i < len(JSONL_RECORDS):
        rec = JSONL_RECORDS[i]
        ecg = rec.get('ecg', [])
        eeg = rec.get('eeg', [])
        ecg_preview = '|'.join([f"{x:.3f}" for x in ecg[:20]]) if ecg else ''
        eeg_preview = '|'.join([f"{x:.3f}" for x in eeg[:20]]) if eeg else ''
        row.setdefault('ecg_preview', ecg_preview)
        row.setdefault('eeg_preview', eeg_preview)
        row['metrics'] = rec.get('metrics', {})
        row['explanation'] = rec.get('explanation', {})
    return jsonify(row)

@app.route('/label', methods=['POST'])
def label():
    j = request.get_json()
    MANUAL_OUT.parent.mkdir(exist_ok=True)
    write_header = not MANUAL_OUT.exists()
    with open(MANUAL_OUT, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(['idx','label'])
        w.writerow([j.get('idx'), j.get('label')])
    return ('',204)

if __name__ == '__main__':
    app.run(port=5001)
