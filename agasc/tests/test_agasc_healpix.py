# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test functionality related to HEALpix ordering of AGASC files.

This uses AGASC 1.8 which is the first release to use such ordering.
"""
import numpy as np
import pytest
import tables
from astropy.table import Table
from ska_helpers.utils import temp_env_var

import agasc
from agasc.healpix import get_healpix_info


# Look for any AGASC 1.8 files in the default AGASC dir, potentially including release
# candidate files. This is to allow testing prior to the final release of AGASC 1.8.
def get_agasc_1p8():
    root = "proseco_agasc"
    agasc_files = agasc.default_agasc_dir().glob(f"{root}_1p8*.h5")
    paths = {path.name: path for path in agasc_files}
    if paths:
        if f"{root}_1p8.h5" in paths:
            agasc_file = paths[f"{root}_1p8.h5"]
        else:
            # Take the last one alphabetically which works for rc < 10
            name = sorted(paths)[-1]
            agasc_file = paths[name]
    else:
        agasc_file = None
    return agasc_file


AGASC_FILE_1P8 = get_agasc_1p8()

if AGASC_FILE_1P8 is None:
    pytest.skip("No proseco_agasc 1.8 file found", allow_module_level=True)


def test_healpix_index():
    """
    Test that the healpix index table is present and has the right number of rows.
    """
    healpix_index_map, nside = get_healpix_info(AGASC_FILE_1P8)
    assert len(healpix_index_map) == 12 * nside**2


ras = np.linspace(0, 180, 10)
decs = np.linspace(-40, 40, 10)
ras = [0]
decs = [-40]


@pytest.mark.parametrize("ra, dec", zip(ras, decs))
def test_get_agasc_cone(ra, dec):
    stars1p7 = agasc.get_agasc_cone(
        ra,
        dec,
        radius=0.2,
        agasc_file=agasc.default_agasc_dir() / "proseco_agasc_1p7.h5",
        date="1997:001",
    )
    stars1p8 = agasc.get_agasc_cone(
        ra, dec, radius=0.2, agasc_file=AGASC_FILE_1P8, date="1997:001"
    )
    assert np.all(stars1p7["AGASC_ID"] == stars1p8["AGASC_ID"])
