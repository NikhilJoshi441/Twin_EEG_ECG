from playwright.sync_api import sync_playwright
import json
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(2000)
    def eval_js(expr):
        try:
            return page.evaluate(f'() => {expr}')
        except Exception as e:
            return {'error': str(e)}
    out={
        'alerts_file_loaded': eval_js('!!window.__alerts_file_loaded'),
        'alerts_registered': eval_js('!!window.__alerts_registered'),
        'alerts_do_init': eval_js('typeof window.__alerts_do_init === "function"'),
        'alerts_init_exists': eval_js('typeof window.__alerts_init === "function"'),
        'alerts_loader': eval_js('window.__alerts_loader || null'),
        'pageErrors': eval_js('window.__pageErrors || []'),
        'lifecycle_flags': eval_js('({fileLoaded:!!window.__alerts_file_loaded, initCalled:!!window.__alerts_init_called, loadCalled:!!window.__loadAlerts_called, initCompleted:!!window.__alerts_init_completed})'),
        'dom_ready': eval_js('document.readyState'),
    }
    print(json.dumps(out, indent=2))
    b.close()
