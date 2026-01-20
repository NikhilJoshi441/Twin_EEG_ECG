from playwright.sync_api import sync_playwright
import json, time
URL = 'http://localhost:5000/alerts'

def run():
    out = {"console": [], "requests": [], "rows": 0}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        # capture console
        page = context.new_page()
        page.on('console', lambda msg: out['console'].append({'type': msg.type, 'text': msg.text}))
        page.on('response', lambda r: out['requests'].append({'url': r.url, 'status': r.status}))
        page.on('requestfailed', lambda req: out['requests'].append({'url': req.url, 'failed': True}))
        # instrument fetch before any scripts run
        context.add_init_script("""
            (function(){
                window.__fetch_log = [];
                const _orig = window.fetch;
                window.fetch = function(input, init){
                    const url = (typeof input === 'string') ? input : (input && input.url) || '';
                    return _orig.apply(this, arguments).then(function(resp){
                        try { resp.clone().text().then(b=>{ window.__fetch_log.push({url:url, status: resp.status, len: b? b.length:0, t: new Date().toISOString()}); }).catch(()=>{}); } catch(e){}
                        return resp;
                    });
                };
            })();
        """)
        page.goto(URL, timeout=30000)
        # allow time for scripts to run; interact visually for a moment
        time.sleep(4)
        try:
            out['rows'] = page.evaluate("() => document.querySelectorAll('#alertsTable tbody tr').length")
        except Exception:
            out['rows'] = 0
        try:
            out['inpage_fetch_log'] = page.evaluate('() => window.__fetch_log || []')
        except Exception:
            out['inpage_fetch_log'] = []
        try:
            out['lifecycle_flags'] = page.evaluate("() => ({ fileLoaded: !!window.__alerts_file_loaded, registered: !!window.__alerts_registered, initCalled: !!window.__alerts_init_called, initCompleted: !!window.__alerts_init_completed, loadCalled: !!window.__loadAlerts_called })")
        except Exception:
            out['lifecycle_flags'] = None
        try:
            out['pageErrors'] = page.evaluate('() => window.__pageErrors || []')
        except Exception:
            out['pageErrors'] = None
        # save screenshot
        try:
            page.screenshot(path='alerts_headed.png', full_page=True)
            out['screenshot'] = 'alerts_headed.png'
        except Exception as e:
            out['screenshot_error'] = str(e)
        print(json.dumps(out, indent=2))
        # keep the browser open for interactive inspection; wait for user to continue
        try:
            input('Interactive headed session open. Inspect the browser, then press Enter here to continue and close the browser...')
        except Exception:
            pass
        browser.close()

if __name__ == '__main__':
    run()
