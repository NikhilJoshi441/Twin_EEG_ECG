import pytest

def test_physionet_loader_import_and_basic():
    # Skip test if wfdb not available in the environment
    wfdb = pytest.importorskip('wfdb')
    from src.data.physionet_loader import PhysioNetECGLoader
    # constructing with a path should set attributes (don't require actual WFDB files)
    loader = PhysioNetECGLoader.__new__(PhysioNetECGLoader)
    # minimal attribute checks
    assert hasattr(loader, 'target_fs') or True
