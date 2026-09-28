import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_packages_import_from_src():
    for module in (
        'configs',
        'dataset',
        'models',
        'utils.losses',
        'utils.metrics',
        'utils.optim',
        'utils.tools',
    ):
        assert importlib.import_module(module)


def test_runtime_directories_are_separated():
    assert (ROOT / 'src').is_dir()
    assert (ROOT / 'tests').is_dir()
    assert (ROOT / 'config').is_dir()
    assert (ROOT / 'data').is_dir()
    assert not (ROOT / 'losses').exists()
    assert not (ROOT / 'metrics').exists()
    assert not (ROOT / 'optim').exists()
    assert not (ROOT / 'tools').exists()
