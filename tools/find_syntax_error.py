from playwright.sync_api import sync_playwright
import json
URL='http://localhost:5000/alerts'
OUT='find_error.json'

with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    t = page.evaluate("() => fetch('/static/js/alerts.js').then(r=>r.text())")
    if not t:
        print(json.dumps({'error':'failed_fetch'}))
        b.close()
        raise SystemExit(1)
    L = len(t)
    # quick check
    def test(n):
        return page.evaluate(f"(function(){{ try{{ new Function(window.__t.slice(0,{n})); return {{ok:true}}; }}catch(e){{ return {{ok:false, err:String(e)}}; }} }})()")
    # store script on window to avoid re-supplying
    page.evaluate(f"() => (window.__t = {json.dumps(t)})")
    low = 0
    high = L
    # ensure low ok
    ok_low = test(low)
    ok_high = test(high)
    if not ok_low.get('ok', False):
        print(json.dumps({'error':'empty_not_ok', 'detail': ok_low}))
        b.close()
        raise SystemExit(1)
    if ok_high.get('ok', False):
        # full script parses fine as function body -- report that
        res = {'ok': True, 'note': 'full_parses_as_function_body', 'len': L}
        print(json.dumps(res, indent=2))
        b.close()
        raise SystemExit(0)
    # binary search
    while high - low > 1:
        mid = (low + high) // 2
        r = test(mid)
        if r.get('ok'):
            low = mid
        else:
            high = mid
    # get error at high
    err_r = test(high)
    snippet = t[max(0, high-80):min(L, high+80)]
    res = {'ok': False, 'len': L, 'bad_index': high, 'error': err_r.get('err'), 'snippet': snippet}
    print(json.dumps(res, indent=2))
    b.close()
