from playwright.sync_api import sync_playwright
URL='http://localhost:5000/alerts'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    ctx=b.new_context()
    page=ctx.new_page()
    resp = page.goto(URL, timeout=15000)
    headers = resp.headers if resp else {}
    print(headers)
    b.close()
