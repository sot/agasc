# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

SUPPLEMENT_FILENAME = 'agasc_supplement.h5'


__all__ = ['default_agasc_dir', 'default_agasc_file', 'SUPPLEMENT_FILENAME']


def default_agasc_dir():
    """Path to the AGASC directory.

    This returns the ``AGASC_DIR`` environment variable if defined, otherwise
    ``$SKA/data/agasc``.

    :returns: Path
    """
    if 'AGASC_DIR' in os.environ:
        out = Path(os.environ['AGASC_DIR'])
    else:
        out = Path(os.environ['SKA'], 'data', 'agasc')
    return out


def default_agasc_file():
    """Default main AGASC file.

    :returns: str
    """
    from agasc import get_agasc_filename
    return get_agasc_filename()
