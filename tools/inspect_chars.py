import urllib.request
u='http://localhost:5000/static/js/alerts.js'
r=urllib.request.urlopen(u)
b=r.read()
for i in range(60, 120):
    ch = b[i:i+1]
    try:
        s=ch.decode('utf-8')
    except:
        s='?'
    print(i, repr(s), hex(ord(s)) if s!='?' else ch)
print('\nSNIPPET REPR:')
print(repr(b[40:120]))
