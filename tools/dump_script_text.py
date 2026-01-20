import urllib.request
u='http://localhost:5000/static/js/alerts.js'
r=urllib.request.urlopen(u)
b=r.read()
print('len=', len(b))
print(b[:800].decode('utf-8', errors='replace'))
print('\nREPR:\n', repr(b[:800]))
