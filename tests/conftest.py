import pytest


@pytest.fixture(autouse=True)
def _isolated_verify_cache(tmp_path_factory, monkeypatch):
    """Keep verify="cached" results out of the real ~/.cache/deltatensors."""
    monkeypatch.setenv("DELTATENSORS_CACHE_DIR", str(tmp_path_factory.mktemp("dt-cache")))
