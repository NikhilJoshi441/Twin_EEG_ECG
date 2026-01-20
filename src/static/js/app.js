console.log('app.js loaded');
const socket = io();

// simple rolling buffer plot using Chart.js
function createChart(ctx, label, color) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{ label: label, data: [], borderColor: color, borderWidth: 1, pointRadius: 0, fill: false, tension: 0 }]
    },
    options: {
      animation: false,
      responsive: true,
      scales: { x: { display: false } },
    }
  });
}

const ecgCtx = document.getElementById('ecgChart').getContext('2d');
const eegCtx = document.getElementById('eegChart').getContext('2d');
const ecgChart = createChart(ecgCtx, 'ECG', 'red');
const eegChart = createChart(eegCtx, 'EEG', 'blue');

// metrics panel
const metricsDiv = document.createElement('div');
metricsDiv.id = 'metrics';
metricsDiv.style.position = 'fixed';
metricsDiv.style.right = '12px';
metricsDiv.style.top = '12px';
metricsDiv.style.background = 'rgba(255,255,255,0.9)';
metricsDiv.style.padding = '8px';
metricsDiv.style.border = '1px solid #ddd';
metricsDiv.style.fontSize = '12px';
metricsDiv.style.zIndex = 1000;
document.body.appendChild(metricsDiv);

// threat / AI indicator
const threatDiv = document.createElement('div');
threatDiv.id = 'threat';
threatDiv.style.position = 'fixed';
threatDiv.style.right = '12px';
threatDiv.style.top = '180px';
threatDiv.style.width = '160px';
threatDiv.style.height = '36px';
threatDiv.style.background = 'green';
threatDiv.style.color = 'white';
threatDiv.style.display = 'flex';
threatDiv.style.alignItems = 'center';
threatDiv.style.justifyContent = 'center';
threatDiv.style.borderRadius = '4px';
threatDiv.style.fontWeight = '600';
threatDiv.textContent = 'Threat: unknown';
threatDiv.style.zIndex = 1001;
document.body.appendChild(threatDiv);

// recent threat history list
const historyDiv = document.createElement('div');
historyDiv.id = 'threatHistory';
historyDiv.style.position = 'fixed';
historyDiv.style.right = '12px';
historyDiv.style.top = '230px';
historyDiv.style.width = '240px';
historyDiv.style.maxHeight = '360px';
historyDiv.style.overflowY = 'auto';
historyDiv.style.background = 'rgba(255,255,255,0.95)';
historyDiv.style.border = '1px solid #ddd';
historyDiv.style.padding = '8px';
historyDiv.style.fontSize = '12px';
historyDiv.style.zIndex = 1001;
historyDiv.innerHTML = '<strong>Recent Alerts</strong><div id="alertsList"></div>';
document.body.appendChild(historyDiv);

const MAX_POINTS = 500;
let ecgCounter = 0;
let eegCounter = 0;

socket.on('connect', () => {
  console.log('Socket.IO connected');
  // show small client-side note
  const info = document.createElement('div');
  info.style.fontSize = '12px';
  info.style.color = '#333';
  info.textContent = 'Socket connected';
  document.body.insertBefore(info, document.body.firstChild);
});

// wire up controls
document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const autoscale = document.getElementById('autoscale');
  if (startBtn) startBtn.onclick = () => socket.emit('control', { action: 'start' });
  if (stopBtn) stopBtn.onclick = () => socket.emit('control', { action: 'stop' });
  const applyBtn = document.getElementById('applyThresh');
  const highInput = document.getElementById('highThresh');
  const medInput = document.getElementById('mediumThresh');
  if (applyBtn) applyBtn.onclick = () => {
    const high = parseFloat(highInput.value);
    const medium = parseFloat(medInput.value);
    socket.emit('control', { thresholds: { high: high, medium: medium } });
  };
  if (autoscale) autoscale.onchange = () => {
    const enabled = autoscale.checked;
    // simple autoscale: set y-axis min/max to auto or fixed
    [ecgChart, eegChart].forEach(ch => {
      if (enabled) {
        ch.options.scales.y = { suggestedMin: undefined, suggestedMax: undefined };
      } else {
        ch.options.scales.y = { min: -2, max: 2 };
      }
      ch.update('none');
    });
  };
});

socket.on('threat_history', (h) => {
  try {
    const list = document.getElementById('alertsList');
    if (!list) return;
    list.innerHTML = '';
    const slice = Array.isArray(h) ? h.slice(-10).reverse() : [];
    for (const item of slice) {
      const el = document.createElement('div');
      el.style.marginTop = '6px';
      el.style.cursor = 'pointer';
      el.style.padding = '6px';
      el.style.borderBottom = '1px solid #f0f0f0';
      const header = document.createElement('div');
      header.style.fontWeight = '600';
      header.textContent = `${item.level.toUpperCase()} ${(item.score*100).toFixed(0)}%`;
      const sub = document.createElement('div');
      sub.style.fontSize = '11px';
      sub.style.color = '#333';
      sub.textContent = new Date(item.ts).toLocaleTimeString();
      el.appendChild(header);
      el.appendChild(sub);
      // hidden details container
      const details = document.createElement('pre');
      details.style.display = 'none';
      details.style.whiteSpace = 'pre-wrap';
      details.style.marginTop = '6px';
      details.style.fontSize = '11px';
      try {
        details.textContent = JSON.stringify(item.explanation || item, null, 2);
      } catch (e) { details.textContent = String(item.explanation || ''); }
      el.appendChild(details);
      el.onclick = () => {
        details.style.display = (details.style.display === 'none') ? 'block' : 'none';
      };
      // detail button to request server-side timeline + raw signals
      const btn = document.createElement('button');
      btn.textContent = 'Open Detail';
      btn.style.marginLeft = '8px';
      btn.onclick = (ev) => {
        ev.stopPropagation();
        socket.emit('request_alert_detail', { ts: item.ts, window_s: 2.0 });
      };
      el.appendChild(btn);
      list.appendChild(el);
    }
  } catch (e) { console.warn('history render error', e); }
});

socket.on('thresholds', (t) => {
  try {
    const highInput = document.getElementById('highThresh');
    const medInput = document.getElementById('mediumThresh');
    if (highInput && typeof t.high === 'number') highInput.value = t.high.toFixed(2);
    if (medInput && typeof t.medium === 'number') medInput.value = t.medium.toFixed(2);
  } catch (e) { }
});

socket.on('alert_detail', (d) => {
  try {
    if (d.error) { alert('Detail error: ' + d.error); return; }
    // create modal
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.left = '0';
    modal.style.top = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.background = 'rgba(0,0,0,0.5)';
    modal.style.zIndex = 2000;
    const box = document.createElement('div');
    box.style.position = 'absolute';
    box.style.left = '50%';
    box.style.top = '50%';
    box.style.transform = 'translate(-50%,-50%)';
    box.style.background = '#fff';
    box.style.padding = '12px';
    box.style.borderRadius = '6px';
    box.style.width = '840px';
    box.style.maxHeight = '80%';
    box.style.overflow = 'auto';
    const close = document.createElement('button');
    close.textContent = 'Close';
    close.onclick = () => { document.body.removeChild(modal); };
    box.appendChild(close);

    const hdr = document.createElement('h3');
    hdr.textContent = `Alert ${d.ts}`;
    box.appendChild(hdr);

    const expl = document.createElement('div');
    expl.innerHTML = `<strong>Summary:</strong> ${d.explanation_text || ''}`;
    box.appendChild(expl);

    const metricsPre = document.createElement('pre');
    metricsPre.textContent = JSON.stringify(d.metrics || {}, null, 2);
    box.appendChild(metricsPre);

    // add canvases for raw signals
    const ecgCanvas = document.createElement('canvas'); ecgCanvas.width = 800; ecgCanvas.height = 120;
    const eegCanvas = document.createElement('canvas'); eegCanvas.width = 800; eegCanvas.height = 120;
    box.appendChild(document.createTextNode('ECG raw signal (near alert)'));
    box.appendChild(ecgCanvas);
    box.appendChild(document.createTextNode('EEG raw signal (near alert)'));
    box.appendChild(eegCanvas);

    modal.appendChild(box);
    document.body.appendChild(modal);

    // plot signals if present using Chart.js
    try {
        const ecgCtx = ecgCanvas.getContext('2d');
        const eegCtx = eegCanvas.getContext('2d');

        // small plugin to draw a vertical marker at r_peak_pos if set on chart
        const rPeakPlugin = {
          id: 'rPeak',
          afterDraw: (chart) => {
            try {
              const pos = chart.$rPeakPos;
              if (pos == null) return;
              const ctx = chart.ctx;
              const xScale = chart.scales.x;
              const x = xScale.getPixelForValue(pos);
              ctx.save();
              ctx.beginPath();
              ctx.strokeStyle = 'rgba(200,20,20,0.9)';
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 3]);
              ctx.moveTo(x, chart.chartArea.top);
              ctx.lineTo(x, chart.chartArea.bottom);
              ctx.stroke();
              ctx.restore();
            } catch (e) { /* ignore */ }
          }
        };

        const make = (ctx, label, data, rpos) => {
          const cfg = {
            type: 'line',
            data: { labels: data.map((_,i)=>i), datasets:[{label:label, data:data, borderColor:'red', pointRadius:0, borderWidth:1, fill:false}] },
            options:{
              animation:false,
              plugins:{legend:{display:false}, zoom:{pan:{enabled:true, mode:'x'}, zoom:{wheel:{enabled:true}, pinch:{enabled:true}, mode:'x'}}},
              scales:{x:{display:false}}
            },
            plugins: [rPeakPlugin]
          };
          const c = new Chart(ctx, cfg);
          if (typeof rpos === 'number') c.$rPeakPos = rpos;
          return c;
        };

        if (Array.isArray(d.ecg) && d.ecg.length>0) make(ecgCtx, 'ECG', d.ecg.slice(0, 400), d.r_peak_pos);
        if (Array.isArray(d.eeg) && d.eeg.length>0) make(eegCtx, 'EEG', d.eeg.slice(0, 400));
    } catch (e) { console.warn('plot error', e); }

  } catch (e) { console.warn('alert_detail handler error', e); }
});

socket.on('signal', (msg) => {
  // msg: { ecg: [..], eeg: [..] }
  if (msg.ecg) {
    // console.log('ECG chunk received, length=', msg.ecg.length);
    for (let v of msg.ecg) {
      ecgChart.data.datasets[0].data.push(v);
      ecgChart.data.labels.push(ecgCounter++);
      if (ecgChart.data.datasets[0].data.length > MAX_POINTS) {
        ecgChart.data.datasets[0].data.shift();
        ecgChart.data.labels.shift();
      }
    }
    ecgChart.update('none');
  }
  if (msg.eeg) {
    // console.log('EEG chunk received, length=', msg.eeg.length);
    for (let v of msg.eeg) {
      eegChart.data.datasets[0].data.push(v);
      eegChart.data.labels.push(eegCounter++);
      if (eegChart.data.datasets[0].data.length > MAX_POINTS) {
        eegChart.data.datasets[0].data.shift();
        eegChart.data.labels.shift();
      }
    }
    eegChart.update('none');
  }
});

socket.on('metrics', (m) => {
  // simple render of key metrics (guard for undefined fields)
  const safe = (v, d=0) => (typeof v === 'number' ? v : d);
  metricsDiv.innerHTML = `
    <strong>Metrics</strong><br>
    HR: ${safe(m.avg_hr_bpm,0).toFixed(1)} bpm<br>
    SDNN: ${safe(m.SDNN,0).toFixed(3)} s<br>
    RMSSD: ${safe(m.RMSSD,0).toFixed(3)} s<br>
    pNN50: ${safe(m.pNN50,0).toFixed(1)} %<br>
    LF/HF: ${safe(m.LF_HF,0).toFixed(2)}<br>
    Alpha: ${safe(m.alpha_power,0).toFixed(3)}<br>
    Beta: ${safe(m.beta_power,0).toFixed(3)}<br>
    Entropy: ${safe(m.spectral_entropy,0).toFixed(3)}
  `;
});

socket.on('threat', (t) => {
  // t: { score: 0-1, level: 'low'|'medium'|'high', color }
  try {
    const score = (typeof t.score === 'number') ? t.score : 0.0;
    const level = t.level || 'unknown';
    const color = t.color || (score > 0.7 ? 'red' : score > 0.35 ? 'orange' : 'green');
    threatDiv.style.background = color;
    threatDiv.textContent = `Threat: ${level.toUpperCase()} (${(score*100).toFixed(0)}%)`;
    // subtle flash on high
    if (level === 'high') {
      threatDiv.style.boxShadow = '0 0 12px rgba(255,0,0,0.6)';
      setTimeout(() => { threatDiv.style.boxShadow = ''; }, 600);
    }
  } catch (e) {
    console.warn('Error rendering threat', e);
  }
});
