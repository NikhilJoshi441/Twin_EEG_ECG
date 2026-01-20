from playwright.sync_api import sync_playwright
import json

URL = 'http://localhost:5000/alerts'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(1000)
    flags = page.evaluate("() => ({ fileLoaded: !!window.__alerts_file_loaded, initCalled: !!window.__alerts_init_called, loadCalled: !!window.__loadAlerts_called, initCompleted: !!window.__alerts_init_completed, lastDebug: window.__lastAlertsDebug || null, scripts: Array.from(document.querySelectorAll('script')).map(s=>({src: s.src, type: s.type, async: s.async, defer: s.defer, textLen: s.textContent ? s.textContent.length : 0})) })")
    print(json.dumps(flags, indent=2))
    browser.close()
