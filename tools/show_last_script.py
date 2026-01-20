from playwright.sync_api import sync_playwright
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(1000)
    info = page.evaluate("() => { const s = Array.from(document.querySelectorAll('script')); const last = s[s.length-1]; return {count: s.length, last_text_head: last ? (last.textContent? last.textContent.slice(0,400) : '') : ''}; }")
    print(info)
    b.close()
