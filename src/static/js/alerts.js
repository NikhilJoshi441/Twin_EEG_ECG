// Very-early execution beacon: helps detect whether the external file actually executes.
try {
  const payload = JSON.stringify({ event: 'alerts_file_executed', ts: new Date().toISOString() });
  if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
    try { navigator.sendBeacon('/api/client_debug', payload); } catch (e) { }
  } else {
    try { fetch('/api/client_debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload }).catch(() => { }); } catch (e) { }
  }
} catch (e) { }
console.log('alerts.js loaded');
try { window.__alerts_file_loaded = true; } catch (e) { }
try { window.__alerts_registered = true; } catch (e) { }
// Defer execution until DOM ready and add defensive error handling
function __alerts_init() {
  try {
    // beacon the server that init began (helps diagnose environments where script appears not to run)
    try {
      fetch('/api/client_debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ event: 'alerts_init_attempt', ts: new Date().toISOString() }) }).catch(() => { });
    } catch (e) { }
    try { window.__alerts_init_called = true; } catch (e) { }
    const socket = (typeof io === 'function') ? io() : null;

    async function loadAlerts() {
      try {
        try { window.__loadAlerts_called = true; } catch (e) { }
        console.log('loadAlerts: start');
        const levelEl = document.getElementById('levelFilter');
        const modelEl = document.getElementById('modelFilter');
        if (!levelEl || !modelEl) {
          console.warn('loadAlerts: filter elements missing');
          return;
        }
        const level = levelEl.value;
        const model = modelEl.value;
        const q = new URLSearchParams(); if (level) q.set('level', level); if (model) q.set('model', model);
        const res = await fetch('/api/alerts?' + q.toString());
        if (!res.ok) throw new Error('Fetch failed: ' + res.status);
        const obj = await res.json();
        console.log('loadAlerts raw obj:', obj);
        const tbody = document.querySelector('#alertsTable tbody');
        console.log('loadAlerts: alertsTable tbody present?', !!tbody);
        if (!tbody) throw new Error('alerts table body not found');
        const list = obj.alerts || [];
        console.log('loadAlerts: got', list.length, 'alerts');
        // expose debug info to external test harnesses
        try { window.__lastAlertsDebug = { raw: obj, list_len: list.length, ts: new Date().toISOString() }; } catch (e) { }
        tbody.innerHTML = '';
        for (const a of list) {
          try {
            const tr = document.createElement('tr');
            const score = (a && a.threat && typeof a.threat.score === 'number') ? a.threat.score : 0.0;
            const modelName = (a && a.threat && a.threat.model) || (a && a.explanation && a.explanation.model) || '';
            const explText = a && a.explanation ? JSON.stringify(a.explanation) : '';
            const pred = (a.threat && a.threat.predicted_class) || (a.explanation && a.explanation.predicted_class) || '';
            tr.innerHTML = `<td>${a.timestamp}</td><td>${(a.threat || {}).level || ''}</td><td>${score.toFixed(2)}</td><td>${modelName}</td><td>${pred}</td><td>${explText}</td><td></td>`;
            const btn = document.createElement('button'); btn.textContent = 'Open Detail';
            btn.addEventListener('click', (ev) => { ev.preventDefault(); if (socket) socket.emit('request_alert_detail', { ts: a.timestamp, window_s: 3.0 }); else alert('Socket not available for detail request'); });
            // actions cell is last
            tr.cells[tr.cells.length - 1].appendChild(btn);
            tbody.appendChild(tr);
          } catch (rowErr) {
            console.error('Error rendering alert row', rowErr, a);
          }
        }
        try { if (window.__lastAlertsDebug) window.__lastAlertsDebug.rendered = tbody.querySelectorAll('tr').length; } catch (e) { }
        try { if (window.__lastAlertsDebug && window.__lastAlertsDebug.list_len > 0 && window.__lastAlertsDebug.rendered === 0) window.__lastAlertsDebug.error = 'no_rows_after_render'; } catch (e) { }
      } catch (err) {
        console.error('loadAlerts failed', err);
      }
    }

    const applyBtn = document.getElementById('apply');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        try {
          const el = document.getElementById('exportLink'); const q = new URLSearchParams(); const level = document.getElementById('levelFilter').value; const model = document.getElementById('modelFilter').value; if (level) q.set('level', level); if (model) q.set('model', model); if (el) el.href = '/api/alerts/export?' + q.toString(); loadAlerts();
        } catch (e) { console.error('apply handler error', e); }
      });
    }

    if (socket) {
      socket.on('alert_detail', (d) => {
        try {
          if (d.error) { alert('Detail error: ' + d.error); return; }
          // simple modal showing json + mini plots
          const modal = document.createElement('div'); modal.style.position = 'fixed'; modal.style.top = '0'; modal.style.left = '0'; modal.style.width = '100%'; modal.style.height = '100%'; modal.style.background = 'rgba(0,0,0,0.5)'; modal.style.zIndex = 3000;
          const box = document.createElement('div'); box.style.background = '#fff'; box.style.padding = '12px'; box.style.margin = '40px auto'; box.style.width = '90%'; box.style.maxHeight = '80%'; box.style.overflow = 'auto';
          const close = document.createElement('button'); close.textContent = 'Close'; close.onclick = () => document.body.removeChild(modal);
          box.appendChild(close);
          const hdr = document.createElement('h3'); hdr.textContent = 'Alert ' + d.ts; box.appendChild(hdr);
          const pre = document.createElement('pre'); pre.textContent = JSON.stringify(d, null, 2); box.appendChild(pre);
          // if class probabilities exist, render them in a small table
          const probs = (d.explanation && d.explanation.class_probs) || (d.explanation && d.explanation.class_probs === 0 ? d.explanation.class_probs : null);
          if (d.explanation && d.explanation.class_probs && typeof d.explanation.class_probs === 'object') {
            const tbl = document.createElement('table'); tbl.style.marginTop = '8px'; tbl.style.borderCollapse = 'collapse';
            for (const k of Object.keys(d.explanation.class_probs)) {
              const row = document.createElement('tr');
              const ktd = document.createElement('td'); ktd.textContent = k; ktd.style.padding = '4px 8px'; ktd.style.fontWeight = '600';
              const vtd = document.createElement('td'); vtd.textContent = (d.explanation.class_probs[k]||0).toFixed(3); vtd.style.padding = '4px 8px';
              row.appendChild(ktd); row.appendChild(vtd); tbl.appendChild(row);
            }
            box.appendChild(tbl);
          }
          // small ECG canvas
          if (Array.isArray(d.ecg) && d.ecg.length > 0) {
            const c = document.createElement('canvas'); c.width = 1000; c.height = 200; box.appendChild(c);
            try {
              const ctx = c.getContext('2d');
              const chart = new Chart(ctx, { type: 'line', data: { labels: d.ecg.map((_, i) => i), datasets: [{ data: d.ecg, borderColor: 'red', pointRadius: 0 }] }, options: { animation: false, plugins: { legend: { display: false }, zoom: { pan: { enabled: true, mode: 'x' }, zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' } } } } });
              if (typeof d.r_peak_pos === 'number') {
                // draw a vertical marker
                chart.$rPeakPos = d.r_peak_pos;
                const plugin = { id: 'rpeak', afterDraw: (chart) => { try { const pos = chart.$rPeakPos; if (pos == null) return; const x = chart.scales.x.getPixelForValue(pos); const ctx = chart.ctx; ctx.save(); ctx.strokeStyle = 'rgba(200,0,0,0.9)'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x, chart.chartArea.top); ctx.lineTo(x, chart.chartArea.bottom); ctx.stroke(); ctx.restore(); } catch (e) { } } };
                chart.options.plugins = chart.options.plugins || {}; chart.plugins = chart.plugins || []; chart.plugins.push(plugin); chart.update();
              }
            } catch (e) { console.warn('plot err', e); }
          }
          modal.appendChild(box); document.body.appendChild(modal);
        } catch (e) { console.error('alert_detail handler error', e); }
      });
    } else {
      console.warn('Socket.IO client not available; detail view will be disabled');
    }

    // initial load
    loadAlerts().then(() => { try { window.__alerts_init_completed = true; } catch (e) { } }).catch(() => { });

  } catch (e) {
    console.error('alerts.js initialization error', e);
  }
}

// expose init hook for external loader to call reliably
try { window.__alerts_do_init = __alerts_init; } catch (e) { }
// preserve prior behavior: attempt to initialize now if DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', __alerts_init);
} else {
  // call asynchronously so loaders that attach handlers have time
  setTimeout(__alerts_init, 0);
}
