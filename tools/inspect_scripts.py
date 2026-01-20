from playwright.sync_api import sync_playwright
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(1000)
    scripts = page.evaluate("() => Array.from(document.querySelectorAll('script')).map(s=>({src: s.src || '', type: s.type || '', async: s.async, textLen: s.textContent? s.textContent.length : 0}))")
    print(scripts)
    b.close()
