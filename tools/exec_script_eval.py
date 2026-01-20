from playwright.sync_api import sync_playwright
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(1000)
    res = page.evaluate("() => fetch('/static/js/alerts.js').then(r=>r.text()).then(t=>{ window.__fetchedScript = t; return {ok:true, len:t.length, head: t.slice(0,400), tail: t.slice(-200)} }).catch(e=>({ok:false, err:String(e)}))")
    print(res)
    # attempt eval separately to capture error
    res2 = page.evaluate("() => { try { eval(window.__fetchedScript || ''); return {ok:true}; } catch(e) { return {ok:false, err:String(e)} } }")
    print('eval-attempt:', res2)
    flags = page.evaluate("() => ({fileLoaded:!!window.__alerts_file_loaded, registered:!!window.__alerts_registered, do_init: typeof window.__alerts_do_init === 'function'})")
    print(flags)
    b.close()
