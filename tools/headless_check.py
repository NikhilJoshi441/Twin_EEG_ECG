from playwright.sync_api import sync_playwright
import json, time, sys

URL = 'http://localhost:5000/alerts'


def run():
    out = {"console": [], "requests": [], "rows": 0, "alerts_preview": []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            # inject a script before any page script runs to instrument window.fetch
            context.add_init_script("""
                (function(){
                    window.__fetch_log = [];
                    const _orig = window.fetch;
                    window.fetch = function(input, init) {
                        const url = (typeof input === 'string') ? input : (input && input.url) || '';
                        const start = Date.now();
                        return _orig.apply(this, arguments).then(function(resp){
                            try {
                                const clone = resp.clone();
                                clone.text().then(function(body){
                                    window.__fetch_log.push({url: url, status: resp.status, ok: resp.ok, len: body ? body.length : 0, timestamp: new Date().toISOString()});
                                }).catch(()=>{
                                    window.__fetch_log.push({url: url, status: resp.status, ok: resp.ok, len: null, timestamp: new Date().toISOString()});
                                });
                            } catch(e) {
                                try { window.__fetch_log.push({url: url, status: resp.status, ok: resp.ok, err: String(e), timestamp: new Date().toISOString()}); } catch(_){}
                            }
                            return resp;
                        }).catch(function(err){
                            try { window.__fetch_log.push({url: url, failed: true, err: String(err), timestamp: new Date().toISOString()}); } catch(_){}
                            throw err;
                        });
                    };
                })();
            """)
            page = context.new_page()

            def on_console(msg):
                try:
                    out['console'].append({'type': msg.type, 'text': msg.text})
                except Exception:
                    out['console'].append({'type': 'unknown', 'text': str(msg)})

            def on_response(resp):
                try:
                    out['requests'].append({'url': resp.url, 'status': resp.status})
                except Exception:
                    out['requests'].append({'url': getattr(resp, 'url', None)})

            def on_request_failed(req):
                out['requests'].append({'url': req.url, 'failed': True})

            page.on('console', on_console)
            page.on('response', on_response)
            page.on('requestfailed', on_request_failed)

            page.goto(URL, timeout=15000)
            # wait for scripts and fetches to run
            time.sleep(3)

            # capture row count
            try:
                rows = page.evaluate("() => document.querySelectorAll('#alertsTable tbody tr').length")
                out['rows'] = rows
            except Exception:
                out['rows'] = 0

            # fetch the in-page fetch log recorded by our init script
            try:
                fetch_log = page.evaluate('() => window.__fetch_log || []')
                out['inpage_fetch_log'] = fetch_log
            except Exception:
                out['inpage_fetch_log'] = []

            # fetch any debug info the page may expose
            try:
                alerts_debug = page.evaluate('() => window.__lastAlertsDebug || null')
                out['alerts_debug'] = alerts_debug
            except Exception:
                out['alerts_debug'] = None

            # capture DOM readiness and presence of expected filter elements
            try:
                ready = page.evaluate('() => ({ ready: document.readyState, hasLevel: !!document.getElementById(\'levelFilter\'), hasModel: !!document.getElementById(\'modelFilter\') })')
                out['dom_state'] = ready
            except Exception:
                out['dom_state'] = None

            # capture lifecycle flags set by alerts.js (if any)
            try:
                lifecycle = page.evaluate("() => ({ fileLoaded: !!window.__alerts_file_loaded, initCalled: !!window.__alerts_init_called, loadCalled: !!window.__loadAlerts_called, initCompleted: !!window.__alerts_init_completed })")
                out['lifecycle_flags'] = lifecycle
            except Exception:
                out['lifecycle_flags'] = None

            # fetch the currently-served alerts.js for inspection (trimmed)
            try:
                script_text = page.evaluate("() => fetch('/static/js/alerts.js').then(r=>r.text()).catch(e=>String(e))")
                if isinstance(script_text, str):
                    out['alerts_js_head'] = script_text[:2000]
                else:
                    out['alerts_js_head'] = None
            except Exception:
                out['alerts_js_head'] = None

            # also attempt a direct fetch from page context to /api/alerts to see response
            try:
                direct = page.evaluate("() => fetch('/api/alerts').then(r=>r.json()).catch(e=>({error:String(e)}))")
                out['page_api_payload'] = direct
                # also provide a short summary
                try:
                    out['direct_fetch_api_alerts'] = {'ok': True, 'count': direct.get('count') if isinstance(direct, dict) else None}
                except Exception:
                    out['direct_fetch_api_alerts'] = {'ok': True}
            except Exception as e:
                out['page_api_payload'] = {'error': str(e)}
                out['direct_fetch_api_alerts'] = {'error': str(e)}

            try:
                alerts = page.evaluate("() => { const t = document.querySelector('#alertsTable tbody'); if (!t) return []; return Array.from(t.querySelectorAll('tr')).map(r=>({timestamp: r.cells[0]?.innerText, level: r.cells[1]?.innerText, score: r.cells[2]?.innerText})); }")
                out['alerts_preview'] = alerts[:10]
            except Exception:
                out['alerts_preview'] = []

            # screenshot
            try:
                page.screenshot(path='alerts_page.png', full_page=True)
                out['screenshot'] = 'alerts_page.png'
            except Exception as e:
                out['screenshot_error'] = str(e)

            # As a diagnostic: attempt to fetch and render rows directly from the test harness
            try:
                eval_rows = page.evaluate("() => fetch('/api/alerts').then(r=>r.json()).then(obj=>{const tbody=document.querySelector('#alertsTable tbody'); if(!tbody) return {err:'no_tbody'}; tbody.innerHTML=''; const list=obj.alerts||[]; for(let i=0;i<Math.min(200,list.length);i++){ try{ const a=list[i]; const tr=document.createElement('tr'); const score=(a&&a.threat&&typeof a.threat.score==='number')?a.threat.score:0.0; const modelName=(a&&a.threat&&a.threat.model)||''; const explText=a&&a.explanation?JSON.stringify(a.explanation):''; tr.innerHTML=`<td>${a.timestamp}</td><td>${(a.threat||{}).level||''}</td><td>${score.toFixed(2)}</td><td>${modelName}</td><td>${explText}</td><td></td>`; const btn=document.createElement('button'); btn.textContent='Open Detail'; tr.cells[5].appendChild(btn); tbody.appendChild(tr);}catch(e){} } return {rendered: tbody.querySelectorAll('tr').length, attempted: list.length}; }).catch(e=>({err:String(e)}))")
                out['eval_rendered_rows'] = eval_rows
            except Exception as e:
                out['eval_rendered_rows'] = {'error': str(e)}

            browser.close()
    except Exception as e:
        out['error'] = str(e)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run()
