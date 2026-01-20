from playwright.sync_api import sync_playwright
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    page.goto(URL, timeout=15000)
    page.wait_for_timeout(500)
    page.evaluate("() => { var s = document.createElement('script'); s.textContent = 'window.__inline_test = 123;'; document.body.appendChild(s); }")
    page.wait_for_timeout(200)
    print(page.evaluate('() => window.__inline_test'))
    b.close()
