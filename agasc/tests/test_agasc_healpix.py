# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test functionality related to HEALpix ordering of AGASC files.

This uses AGASC 1.8 which is the first release to use such ordering.
"""
import numpy as np
import pytest

import agasc
from agasc.healpix import get_healpix_info


AGASC_FILES = {}
for root, version in [("proseco_agasc_*", "1p8"), ("agasc_healpix_*", "1p7")]:
    try:
        fn = agasc.get_agasc_filename(root, version=version, allow_rc=True)
    except FileNotFoundError:
        AGASC_FILES[root, version] = None
    else:
        AGASC_FILES[root, version] = fn


@pytest.mark.skipif(
    AGASC_FILES["proseco_agasc_*", "1p8"] is None,
    reason="No AGASC 1.8 file found",
)
def test_healpix_index():
    """
    Test that the healpix index table is present and has the right number of rows.
    """
    agasc_file = AGASC_FILES["proseco_agasc_*", "1p8"]
    healpix_index_map, nside = get_healpix_info(agasc_file)
    assert len(healpix_index_map) == 12 * nside**2


ras = np.linspace(0, 180, 10)
decs = np.linspace(-40, 40, 10)
ras = [0]
decs = [-40]


@pytest.mark.skipif(
    AGASC_FILES["agasc_healpix_*", "1p7"] is None,
    reason="No AGASC 1.7 HEALpix file found",
)
@pytest.mark.parametrize("ra, dec", zip(ras, decs))
def test_get_agasc_cone(ra, dec):
    stars1p7 = agasc.get_agasc_cone(
        ra,
        dec,
        radius=0.2,
        agasc_file=agasc.get_agasc_filename("agasc*", version="1p7"),
        date="2023:001",
    )

    stars1p7_healpix = agasc.get_agasc_cone(
        ra,
        dec,
        radius=0.2,
        agasc_file=AGASC_FILES["agasc_healpix_*", "1p7"],
        date="2023:001",
    )
    assert np.all(stars1p7["AGASC_ID"] == stars1p7_healpix["AGASC_ID"])
