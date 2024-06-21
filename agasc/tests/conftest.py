import pytest

import agasc


@pytest.fixture(autouse=True)
def unset_agasc_hdf5_file_env(monkeypatch):
    """Unset external AGASC_HDF5_FILE environment variable.

    This fixture is before proseco_agasc_1p7 to ensure that the AGASC_HDF5_FILE
    environment variable is not set by default.
    """
    monkeypatch.delenv("AGASC_HDF5_FILE", raising=False)


@pytest.fixture()
def proseco_agasc_1p7(monkeypatch):
    agasc_file = agasc.get_agasc_filename("proseco_agasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)
