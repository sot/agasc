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


@pytest.mark.skipif(
    AGASC_FILES["agasc_healpix_*", "1p7"] is None,
    reason="No AGASC 1.7 HEALpix file found",
)
@pytest.mark.parametrize("ra, dec", zip(ras, decs))
def test_get_functions_dec_order_vs_healpix(ra, dec):
    # AGASC 1p7 is dec-ordered.
    agasc_dec = agasc.get_agasc_filename("agasc*", version="1p7")
    agasc_healpix = agasc.get_agasc_filename("agasc_healpix_*", version="1p7")
    stars_dec = agasc.get_agasc_cone(
        ra, dec, radius=0.2, agasc_file=agasc_dec, date="2023:001"
    )

    stars_healpix = agasc.get_agasc_cone(
        ra, dec, radius=0.2, agasc_file=agasc_healpix, date="2023:001"
    )
    assert np.all(stars_dec == stars_healpix)

    agasc_ids = stars_dec["AGASC_ID"]
    star = agasc.get_star(agasc_ids[0], agasc_file=agasc_dec, date="2023:001")
    star_healpix = agasc.get_star(
        agasc_ids[0], agasc_file=agasc_healpix, date="2023:001"
    )
    assert star == star_healpix

    stars = agasc.get_stars(agasc_ids, agasc_file=agasc_dec, dates="2023:001")
    stars_healpix = agasc.get_stars(
        agasc_ids, agasc_file=agasc_healpix, dates="2023:001"
    )
    assert np.all(stars == stars_healpix)
