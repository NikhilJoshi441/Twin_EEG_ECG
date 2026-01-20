# Smoke test to verify optional imports
import sys
import traceback

def try_import(name):
    try:
        mod = __import__(name)
        v = getattr(mod, '__version__', None)
        print(f'{name}: OK', 'version='+str(v) if v else '')
        return True
    except Exception as e:
        print(f'{name}: FAIL ->', e)
        traceback.print_exc()
        return False

print('Running optional dependency smoke tests...')

ok_wfdb = try_import('wfdb')

# brainflow needs a different import path and has BoardShim
try:
    import brainflow
    from brainflow.board_shim import BoardShim
    v = getattr(brainflow, '__version__', None)
    print('brainflow: OK', 'version='+str(v) if v else '')
    try:
        # Attempt an introspection call that's safe without hardware
        descr = BoardShim.get_board_descr()
        print('BoardShim.get_board_descr(): OK (returned descriptions)')
    except Exception as e:
        print('BoardShim introspection failed (likely no native drivers):', e)
        traceback.print_exc()
except Exception as e:
    print('brainflow: FAIL ->', e)
    traceback.print_exc()

print('\nSmoke tests finished.')
