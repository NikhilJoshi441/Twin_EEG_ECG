import io,sys
p='c:/Users/Nikhil Joshi/OneDrive/Desktop/Twin/src/static/js/alerts.js'
s=open(p,'rb').read()
# decode as utf-8 replacing invalid
text = s.decode('utf-8', 'replace')
# remove U+2028 and U+2029 which break JS parsers in some contexts
text = text.replace('\u2028','\n').replace('\u2029','\n')
# remove other C0 control chars except \t,\n,\r
clean = []
for ch in text:
    o=ord(ch)
    if o<32 and ch not in '\t\n\r':
        continue
    clean.append(ch)
text=''.join(clean)
open(p,'w',encoding='utf-8',newline='\n').write(text)
print('sanitized',p)
