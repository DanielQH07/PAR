import tarfile

from utils.tools.utils import gen_code_archive


def test_code_archive_contains_source_but_not_local_data_or_environment(tmp_path):
    archive = gen_code_archive(tmp_path)

    with tarfile.open(archive, 'r:gz') as tar:
        names = set(tar.getnames())

    assert 'src/train_linear_probe_2027.py' in names
    assert 'config/upar_local.yaml' in names
    assert not any(name.startswith('.venv/') for name in names)
    assert not any(name.startswith('data/') for name in names)
