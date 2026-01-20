import torch
p='src/models/fusion_improved.pth'
try:
    st=torch.load(p,map_location='cpu')
    if isinstance(st,dict):
        if 'model_state_dict' in st:
            st=st['model_state_dict']
        print('keys:', len(st.keys()))
        for k,v in st.items():
            try:
                print(k, getattr(v,'shape',type(v)))
            except Exception:
                print(k, type(v))
    else:
        print('loaded object type', type(st))
except Exception as e:
    print('load error', e)
