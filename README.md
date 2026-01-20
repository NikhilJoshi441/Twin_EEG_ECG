# Twin — EEG + ECG Monitoring

Simple, local software for capturing and reviewing EEG (brain) and ECG (heart) signals with basic automated alerts.

## What it is
- Capture multi‑channel EEG and 1–12 lead ECG recordings.
- Show time‑synced EEG and ECG in a viewer.
- Automatically flag likely events (seizures, arrhythmias) for quick review.

## Key features
- Real‑time streaming viewer with zoom and playback.
- Automated event detection with confidence scores.
- Manual annotation and export (PDF/CSV/EDF).
- Exports and model artifacts stored in `src/models/`.

## What it measures (plain language)
- EEG: brain waves and unusual bursts/spikes — may indicate seizures or brain dysfunction.
- ECG: heart rate, rhythm, and arrhythmias — may indicate AF, tachycardia, pauses, or ischemic changes.
- Fusion: compares EEG + ECG events to help decide if a problem is cardiac or neurological.

## Quickstart (Windows PowerShell)
1. Install Python 3.8+ and create a venv (recommended):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install requirements (if `requirements.txt` exists):
```powershell
py -3 -m pip install -r requirements.txt
```
3. Run the server from the project root:
```powershell
# ensure PYTHONPATH is set so `src` imports work
$env:PYTHONPATH='.'
$env:FUSION_TS_PATH='src/models/fusion_improved_ts.pt' # optional if present
py -3 -m src.server
```
4. Open the viewer in a browser: `http://127.0.0.1:5000/alerts`

## Helpful files
- Server: `src/server.py`
- Frontend: `src/static/alerts.html` and `src/static/js/alerts.js`
- Models: `src/models/` (contains RF/PyTorch artifacts)
- Tools and scripts: `tools/`

## Contributing
- Add issues or PRs for bugs, documentation, or small features.
- For major changes, open an issue first to discuss scope.

## License
Add a license file if you want this repo published publicly. By default, no license is included.

---
If you want, I can also: create a `.gitignore`, commit this `README.md`, and push to your remote. Tell me which step to take next.
# Twin
