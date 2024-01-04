# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test that agasc.get_agasc_cone() gives the same answer as the ASCDS
tool mp_get_agasc.  Note that mp_get_agasc gives incorrect results
for search radius much above 1.4 degrees.  For instance the following
is missing around 50 stars (brighter than 11.5) near the edge of the
search cone::

 % mp_get_agasc -r 0.0 -d 89.9 -w 2.0

Tests here are limited to 1.4 degree radius.

Another difference is that the output precision of ACA_MAG and
ACA_MAG_ERR in mp_get_agasc are limited, so this produces slightly
different results when the faint limit filtering is applied.

Comprehensive test (takes a while):
>>> from test_agasc import interactive_test_agasc
>>> interactive_test_agasc(nsample=100, agasc_file='miniagasc.h5') # local miniagasc
>>> interactive_test_agasc(nsample=100) # installed  (/proj/sot/ska/data/agasc/miniagasc.h5)
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import Ska.Shell
from astropy.io import ascii
from astropy.table import Row, Table

import agasc

os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = "False"

# See if we can get to ASCDS environment and mp_get_agasc
try:
    ascrc_file = "{}/.ascrc".format(os.environ["HOME"])
    assert os.path.exists(ascrc_file)
    ascds_env = Ska.Shell.getenv(
        "source {} -r release".format(ascrc_file), shell="tcsh"
    )
    assert "ASCDS_BIN" in ascds_env
    cmd = "mp_get_agasc -r 10 -d 20 -w 0.01"
    # Run the command to check for bad status (which will throw exception)
    Ska.Shell.run_shell(cmd, shell="bash", env=ascds_env)
    match = re.search(r"agasc([p0-9]+)", ascds_env["ASCDS_AGASC"])
    DS_AGASC_VERSION = match.group(1)
except Exception:
    ascds_env = None
    DS_AGASC_VERSION = None


NO_MAGS_IN_SUPPLEMENT = not any(agasc.get_supplement_table("mags", as_dict=True))
NO_OBS_IN_SUPPLEMENT = not any(agasc.get_supplement_table("obs", as_dict=True))

HAS_KSH = os.path.exists("/bin/ksh")  # dependency of mp_get_agascid

AGASC_COL_DESCR = """
AGASC_ID - a unique long integer used for identification.
RA - double variable expressing right ascension in decimal degrees.
DEC - double variable expressing declination in decimal degrees.
POS_ERR - short integer value of position uncertainty, in milli-arcsec.
POS_CATID - unsigned integer identifying the source of the
EPOCH - float variable identifying the epoch of the ra and dec
PM_RA - short integer variable expressing proper motion in ra in units of
PM_DEC - short integer variable expressing proper motion in dec in units
PM_CATID - unsigned integer identifying the source of the
PLX - short integer variable expressing parallax in units of
PLX_ERR - short integer variable expressing parallax error
PLX_CATID - unsigned integer identifying the source of the
MAG_ACA - float variable expressing the calculated magnitude in the AXAF
MAG_ACA_ERR - short integer expressing the uncertainty of mag_aca in
CLASS - short integer code identifying classification of entry.
MAG - float variable expressing magnitude, in mags.  Spectral
MAG_ERR - short integer value of magnitude uncertainty, in
MAG_BAND - short integer code which identifies the spectral band
MAG_CATID - unsigned integer identifying the source of the
COLOR1 - float variable expressing the cataloged or estimated B-V color,
COLOR1_ERR - short integer expressing the error in color1 in units of
C1_CATID - unsigned integer identifying the source of color1 and
COLOR2 - float variable expressing a different color, in mag.
COLOR2_ERR - short integer expressing the error in color2, iun
C2_CATID - unsigned integer identifying the source of color2 and
RSV1 - Preliminary J-band magnitude of nearby 2MASS extended source.
RSV2 - short integer reserved for future use. Default value of -9999.
RSV3 - unsigned integer reserved for future use. Default value is 0.
VAR - short integer code providing information on known or suspected
VAR_CATID - unsigned integer code identifying the source of VAR
ASPQ1 - short integer spoiler code for aspect stars.
ASPQ2 - short integer proper motion flag.
ASPQ3 - short integer distance (for Tycho-2 stars only) to
ACQQ1 - short integer indicating magnitude difference between the
ACQQ2 - short integer indicating magnitude difference between the
ACQQ3 - short integer indicating magnitude difference between the
ACQQ4 - short integer indicating magnitude difference between the
ACQQ5 - short integer indicating magnitude difference between the
ACQQ6 - short integer indicating magnitude difference between the
XREF_ID1 - long integer which is the star number in the
XREF_ID2 - long integer which maps the entry to that in the PPM.
XREF_ID3 - long integer which maps the entry to that in the Tycho Output
XREF_ID4 - long integer which maps the entry to that in the Tycho Output
XREF_ID5 - long integer which maps the entry to that in a future
RSV4 - short integer reserved for future use.  Default value of -9999.
RSV5 - short integer reserved for future use.  Default value of -9999.
RSV6 - short integer reserved for future use.  Default value of -9999.
"""

AGASC_COLNAMES = [line.split()[0] for line in AGASC_COL_DESCR.strip().splitlines()]


TEST_RADIUS = 0.6  # standard testing radius
TEST_DIR = Path(__file__).parent

# Whether to test DS AGASC vs. agasc package HDF5 files
if DS_AGASC_VERSION is None:
    TEST_ASCDS = False
else:
    try:
        agasc.get_agasc_filename("miniagasc_*", version=DS_AGASC_VERSION)
    except FileNotFoundError:
        TEST_ASCDS = False
    else:
        TEST_ASCDS = True

# Latest full release of miniagasc
MINIAGASC = agasc.get_agasc_filename("miniagasc_*")


def get_ds_agasc_cone(ra, dec):
    cmd = "mp_get_agasc -r {!r} -d {!r} -w {!r}".format(ra, dec, TEST_RADIUS)
    lines = Ska.Shell.tcsh(cmd, env=ascds_env)
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    ok1 = agasc.sphere_dist(ra, dec, dat["RA"], dat["DEC"]) <= TEST_RADIUS
    ok2 = dat["MAG_ACA"] - 3.0 * dat["MAG_ACA_ERR"] / 100.0 < 11.5
    dat = dat[ok1 & ok2]

    if os.environ.get("WRITE_AGASC_TEST_FILES"):
        version = DS_AGASC_VERSION
        test_file = get_test_file(ra, dec, version)
        print(f"\nWriting {test_file} based on mp_get_agasc\n")
        dat.write(test_file, format="fits")

    return dat


def get_test_file(ra, dec, version):
    return TEST_DIR / "data" / f"ref_ra_{ra}_dec_{dec}_{version}.fits.gz"


def get_reference_agasc_values(ra, dec, version="1p7"):
    dat = Table.read(get_test_file(ra, dec, version))
    return dat


RAS = np.hstack([0.0, 180.0, 0.1, 180.0, 275.36])
DECS = np.hstack([89.9, -89.9, 0.0, 0.0, 8.09])
# The (275.36, 8.09) coordinate fails unless date=2000:001 due to
# mp_get_agasc not accounting for proper motion.


@pytest.mark.parametrize("version", ["1p6", "1p7"])
@pytest.mark.parametrize("ra,dec", list(zip(RAS, DECS)))
def test_agasc_conesearch(ra, dec, version):
    """
    Compare results of get_agasc_cone to package reference data stored in
    FITS files.
    """
    try:
        ref_stars = get_reference_agasc_values(ra, dec, version=version)
    except FileNotFoundError:
        if os.environ.get("WRITE_AGASC_TEST_FILES"):
            ref_stars = agasc.get_agasc_cone(
                ra,
                dec,
                radius=TEST_RADIUS,
                agasc_file=agasc.get_agasc_filename("miniagasc_*", version=version),
                date="2000:001",
                fix_color1=False,
            )
            test_file = get_test_file(ra, dec, version)
            print(f"\nWriting {test_file} based on miniagasc\n")
            ref_stars.write(test_file, format="fits")
        pytest.skip("Reference data unavailable")
    else:
        _test_agasc(ra, dec, ref_stars, version)


@pytest.mark.skipif("not TEST_ASCDS")
@pytest.mark.parametrize("ra,dec", list(zip(RAS, DECS)))
def test_against_ds_agasc(ra, dec):
    """
    Compare results of get_agasc_cone to the same star field retrieved from
    the DS command line tool mp_get_agasc.
    """
    ref_stars = get_ds_agasc_cone(ra, dec)
    _test_agasc(ra, dec, ref_stars, version=DS_AGASC_VERSION)


def _test_agasc(ra, dec, ref_stars, version="1p7"):
    agasc_file = agasc.get_agasc_filename("miniagasc_*", version=version)
    stars1 = agasc.get_agasc_cone(
        ra,
        dec,
        radius=TEST_RADIUS,
        agasc_file=agasc_file,
        date="2000:001",
        fix_color1=False,
    )
    stars1.sort("AGASC_ID")

    stars2 = ref_stars.copy()
    stars2.sort("AGASC_ID")

    # First make sure that the common stars are identical
    agasc_ids = set(stars1["AGASC_ID"]).intersection(set(stars2["AGASC_ID"]))
    for agasc_id in agasc_ids:
        star1 = stars1[np.searchsorted(stars1["AGASC_ID"], agasc_id)]
        star2 = stars2[np.searchsorted(stars2["AGASC_ID"], agasc_id)]
        for colname in AGASC_COLNAMES:
            if star1[colname].dtype.kind == "f":
                assert np.all(np.abs(star1[colname] - star2[colname]) < 1e-4)
            else:
                assert star1[colname] == star2[colname]

    # Second make sure that the non-common stars are all just at the edge
    # of the faint mag limit, due to precision loss in mp_get_agasc
    for s1, s2 in ((stars1, stars2), (stars2, stars1)):
        mm1 = set(s1["AGASC_ID"]) - set(s2["AGASC_ID"])
        for agasc_id in mm1:
            idx = np.flatnonzero(s1["AGASC_ID"] == agasc_id)[0]
            star = s1[idx]
            bad_is_star1 = s1 is stars1
            rad = agasc.sphere_dist(ra, dec, star["RA"], star["DEC"])
            adj_mag = star["MAG_ACA"] - 3.0 * star["MAG_ACA_ERR"] / 100.0
            if adj_mag < 11.5 * 0.99:
                # Allow for loss of precision in output of mp_get_agasc
                print("Bad star", agasc_id, rad, adj_mag, bad_is_star1)
                raise AssertionError()


def test_basic():
    star = agasc.get_star(1180612288)  # High-PM star
    assert np.isclose(star["RA"], 219.9053773)
    assert np.isclose(star["DEC"], -60.8371572)

    stars = agasc.get_agasc_cone(star["RA"], star["DEC"], 0.5)
    stars.sort("MAG_ACA")
    agasc_ids = [1180612176, 1180612296, 1180612184, 1180612288, 1180612192]
    mags = [-0.663, -0.576, -0.373, 0.53, 0.667]
    assert np.allclose(stars["AGASC_ID"][:5], agasc_ids)
    assert np.allclose(stars["MAG_ACA"][:5], mags)


def test_get_stars1():
    # First check that get_stars() gives the same as get_star for the scalar case
    star1 = agasc.get_star(1180612288, date="2019:001")
    star2 = agasc.get_stars(1180612288, dates="2019:001")
    assert isinstance(star2, Row)
    for name in star1.colnames:
        assert star1[name] == star2[name]


def test_get_stars2():
    """get_stars() broadcasts ids"""
    star0 = agasc.get_star(1180612288, date="2010:001")
    star1 = agasc.get_star(1180612288, date="2019:001")
    star2 = agasc.get_stars(1180612288, dates=["2010:001", "2019:001"])

    for name in star1.colnames:
        assert star0[name] == star2[0][name]
        assert star1[name] == star2[1][name]


def test_get_stars3():
    agasc_ids = [1180612176, 1180612296, 1180612184, 1180612288, 1180612192]
    mags = [-0.663, -0.576, -0.373, 0.53, 0.667]
    stars = agasc.get_stars(agasc_ids)
    assert np.allclose(stars["MAG_ACA"], mags)


def test_get_stars_many():
    """Test get_stars() with at least GET_STARS_METHOD_THRESHOLD (5000) stars"""
    from agasc import agasc

    stars = agasc.get_agasc_cone(0, 0, radius=0.5)
    agasc_ids = stars["AGASC_ID"]
    stars1 = agasc.get_stars(agasc_ids, dates="2020:001")  # read_where method
    stars2 = agasc.get_stars(
        agasc_ids, dates="2020:001", method_threshold=1
    )  # read entire AGASC

    assert stars1.get_stars_method == "tables_read_where"
    assert stars2.get_stars_method == "read_entire_agasc"

    assert stars1.colnames == stars2.colnames
    for name in stars1.colnames:
        assert np.all(stars1[name] == stars2[name])


def test_float16():
    stars = agasc.get_agasc_cone(np.float16(219.90279), np.float16(-60.83358), 0.015)
    assert stars["AGASC_ID"][0] == 1180612176


def test_proper_motion():
    """
    Test that the filtering in get_agasc_cone correctly expands the initial
    search radius and then does final filtering using PM-corrected positions.
    """
    star = agasc.get_star(1180612288, date="2017:001")  # High-PM star
    radius = 2.0 / 3600  # 5 arcsec

    stars = agasc.get_agasc_cone(star["RA"], star["DEC"], radius, date="2000:001")
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(star["RA"], star["DEC"], radius, date="2017:001")
    assert len(stars) == 0

    stars = agasc.get_agasc_cone(
        star["RA"], star["DEC"], radius, date="2017:001", pm_filter=False
    )
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(
        star["RA_PMCORR"], star["DEC_PMCORR"], radius, date="2017:001"
    )
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(
        star["RA_PMCORR"], star["DEC_PMCORR"], radius, date="2017:001", pm_filter=False
    )
    assert len(stars) == 0


@pytest.mark.parametrize(
    "agasc_id,date,ra_pmcorr,dec_pmcorr,label",
    [
        (
            1180612288,
            "2020:001",
            219.864331,
            -60.831868,
            "high proper motion, epoch 2000",
        ),
        (198451217, "2020:001", 247.892206, 19.276605, "epoch 1982 star"),
        (501219465, "2020:001", 166.998976, 52.822080, "epoch 1984 star"),
    ],
)
def test_add_pmcorr_is_consistent(agasc_id, date, ra_pmcorr, dec_pmcorr, label):
    """
    Check that the proper-motion corrected position is consistent reference/regress values.
    """
    star = agasc.get_star(agasc_id, date=date)
    assert np.isclose(star["RA_PMCORR"], ra_pmcorr, rtol=0, atol=1e-5)
    assert np.isclose(star["DEC_PMCORR"], dec_pmcorr, rtol=0, atol=1e-5)


def mp_get_agascid(agasc_id):
    cmd = "mp_get_agascid {!r}".format(agasc_id)
    lines = Ska.Shell.tcsh(cmd, env=ascds_env)
    lines = [line for line in lines if re.match(r"^\s*\d", line)]
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    return dat


@pytest.mark.skipif("not HAS_KSH")
@pytest.mark.skipif("not TEST_ASCDS")
@pytest.mark.parametrize("ra,dec", list(zip(RAS[:2], DECS[:2])))
def test_agasc_id(ra, dec, radius=0.2, nstar_limit=5):
    agasc_file = agasc.get_agasc_filename("miniagasc_*", version=DS_AGASC_VERSION)

    print("ra, dec =", ra, dec)
    stars = agasc.get_agasc_cone(
        ra, dec, radius=radius, agasc_file=agasc_file, fix_color1=False
    )
    stars.sort("AGASC_ID")

    for agasc_id in stars["AGASC_ID"][:nstar_limit]:
        print("  agasc_id =", agasc_id)
        star1 = agasc.get_star(agasc_id, agasc_file=agasc_file, fix_color1=False)
        star2 = mp_get_agascid(agasc_id)
        for colname in AGASC_COLNAMES:
            if star1[colname].dtype.kind == "f":
                assert np.all(np.allclose(star1[colname], star2[colname]))
            else:
                assert star1[colname] == star2[colname]


def test_proseco_agasc_1p7():
    proseco_file = agasc.get_agasc_filename("proseco_agasc_*", version="1p7")
    mini_file = agasc.get_agasc_filename("miniagasc_*", version="1p7")

    # Stars looking toward galactic center (dense!)
    p_stars = agasc.get_agasc_cone(
        -266, -29, 3, agasc_file=proseco_file, date="2000:001"
    )
    m_stars = agasc.get_agasc_cone(-266, -29, 3, agasc_file=mini_file, date="2000:001")

    # Every miniagasc_1p7 star is in proseco_agasc_1p7
    m_ids = m_stars["AGASC_ID"]
    p_ids = p_stars["AGASC_ID"]
    assert set(m_ids) < set(p_ids)

    # Values are exactly the same
    p_id_map = {p_ids[idx]: idx for idx in np.arange(len(p_ids))}
    for m_star in m_stars:
        m_id = m_star["AGASC_ID"]
        p_star = p_stars[p_id_map[m_id]]
        for name in p_star.colnames:
            assert p_star[name] == m_star[name]


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_supplement_get_agasc_cone():
    ra, dec = 282.53, -0.38  # Obsid 22429 with a couple of color1=1.5 stars
    stars1 = agasc.get_agasc_cone(
        ra, dec, date="2021:001", agasc_file=MINIAGASC, use_supplement=False
    )
    stars2 = agasc.get_agasc_cone(
        ra, dec, date="2021:001", agasc_file=MINIAGASC, use_supplement=True
    )
    ok = stars2["MAG_CATID"] == agasc.MAG_CATID_SUPPLEMENT

    change_names = ["MAG_CATID", "COLOR1", "MAG_ACA", "MAG_ACA_ERR"]
    for name in set(stars1.colnames) - set(change_names):
        assert np.all(stars1[name] == stars2[name])

    assert not np.any(stars1["MAG_CATID"] == agasc.MAG_CATID_SUPPLEMENT)

    # At least 35 stars in this field observed
    assert np.count_nonzero(ok) >= 35

    # At least 7 color=1.5 stars converted to 1.49 (note: total of 23 color=1.5
    # stars in this field)
    assert np.count_nonzero(stars1["COLOR1"][ok] == 1.49) == 0
    assert np.count_nonzero(stars1["COLOR1"][ok] == 1.50) >= 7
    assert np.count_nonzero(stars2["COLOR1"][ok] == 1.49) >= 7
    assert np.count_nonzero(stars2["COLOR1"][ok] == 1.50) == 0

    # For the stars that have updated data for the supplement, confirm they don't
    # have all the same values for MAG_ACA_ERR as the catalog values.  Note this
    # is an integer column.
    assert np.any(stars2["MAG_ACA_ERR"][ok] != stars1["MAG_ACA_ERR"][ok])

    # Similarly, in this set the stars with updated magnitudes are different from
    # the catalog values.
    assert np.all(stars2["MAG_ACA"][ok] != stars1["MAG_ACA"][ok])

    assert np.all(stars2["MAG_ACA_ERR"][~ok] == stars1["MAG_ACA_ERR"][~ok])
    assert np.all(stars2["MAG_ACA"][~ok] == stars1["MAG_ACA"][~ok])


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_supplement_get_star():
    agasc_id = 58720672
    # Also checks that the default is False given the os.environ override for
    # this test file.
    star1 = agasc.get_star(agasc_id, agasc_file=MINIAGASC)
    star2 = agasc.get_star(agasc_id, agasc_file=MINIAGASC, use_supplement=True)
    assert star1["MAG_CATID"] != agasc.MAG_CATID_SUPPLEMENT
    assert star2["MAG_CATID"] == agasc.MAG_CATID_SUPPLEMENT

    assert star1["AGASC_ID"] == star2["AGASC_ID"]

    assert np.isclose(star1["COLOR1"], 1.50)
    assert np.isclose(star2["COLOR1"], 1.49)

    assert star2["MAG_ACA"] != star1["MAG_ACA"]
    assert star2["MAG_ACA_ERR"] != star1["MAG_ACA_ERR"]


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_supplement_get_star_disable_context_manager():
    """Test that disable_supplement_mags context manager works.

    This assumes a global default of AGASC_SUPPLEMENT_ENABLED=False for these
    tests.
    """
    agasc_id = 58720672
    star1 = agasc.get_star(agasc_id, date="2020:001", use_supplement=True)
    with agasc.set_supplement_enabled(True):
        star2 = agasc.get_star(agasc_id, date="2020:001")
    for name in star1.colnames:
        assert star1[name] == star2[name]


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
@agasc.set_supplement_enabled(True)
def test_supplement_get_star_disable_decorator():
    """Test that disable_supplement_mags context manager works"""
    agasc_id = 58720672
    star1 = agasc.get_star(agasc_id, date="2020:001")
    star2 = agasc.get_star(agasc_id, date="2020:001", use_supplement=True)
    for name in star1.colnames:
        assert star1[name] == star2[name]


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_supplement_get_stars():
    agasc_ids = [58720672, 670303120]
    star1 = agasc.get_stars(agasc_ids, agasc_file=MINIAGASC)
    star2 = agasc.get_stars(agasc_ids, agasc_file=MINIAGASC, use_supplement=True)
    assert np.all(star1["MAG_CATID"] != agasc.MAG_CATID_SUPPLEMENT)
    assert np.all(star2["MAG_CATID"] == agasc.MAG_CATID_SUPPLEMENT)

    assert np.all(star1["AGASC_ID"] == star2["AGASC_ID"])

    assert np.allclose(star1["COLOR1"], [1.5, 0.24395067])
    assert np.allclose(star2["COLOR1"], [1.49, 0.24395067])

    assert np.all(star2["MAG_ACA"] != star1["MAG_ACA"])


def test_get_supplement_table_bad():
    bad = agasc.get_supplement_table("bad")
    assert isinstance(bad, Table)
    assert bad.colnames == ["agasc_id", "source"]
    assert len(bad) > 3300
    assert 797847184 in bad["agasc_id"]


def test_get_supplement_table_bad_dict():
    bad = agasc.get_supplement_table("bad", as_dict=True)
    assert isinstance(bad, dict)
    assert len(bad) > 3300
    assert bad[797847184] == {"source": 1}


@agasc.set_supplement_enabled(True)
def test_get_bad_star_with_supplement():
    agasc_id = 797847184
    star = agasc.get_star(agasc_id, use_supplement=True)
    assert star["CLASS"] == agasc.BAD_CLASS_SUPPLEMENT


def test_bad_agasc_supplement_env_var():
    try:
        os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = "asdfasdf"
        with pytest.raises(ValueError, match="env var must be either"):
            agasc.get_star(797847184)
    finally:
        os.environ[agasc.SUPPLEMENT_ENABLED_ENV] = "False"


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_get_supplement_table_mags():
    mags = agasc.get_supplement_table("mags")
    assert isinstance(mags, Table)
    assert 131736 in mags["agasc_id"]
    assert len(mags) > 80000
    assert mags.colnames == ["agasc_id", "mag_aca", "mag_aca_err", "last_obs_time"]


@pytest.mark.skipif(NO_MAGS_IN_SUPPLEMENT, reason="no mags in supplement")
def test_get_supplement_table_mags_dict():
    mags = agasc.get_supplement_table("mags", as_dict=True)
    assert isinstance(mags, dict)
    assert 131736 in mags
    assert len(mags) > 80000
    assert list(mags[131736].keys()) == ["mag_aca", "mag_aca_err", "last_obs_time"]


@pytest.mark.skipif(NO_OBS_IN_SUPPLEMENT, reason="no obs in supplement")
def test_get_supplement_table_obs():
    obs = agasc.get_supplement_table("obs")
    assert isinstance(obs, Table)
    assert obs.colnames == [
        "mp_starcat_time",
        "agasc_id",
        "obsid",
        "status",
        "comments",
    ]


@pytest.mark.skipif(NO_OBS_IN_SUPPLEMENT, reason="no obs in supplement")
def test_get_supplement_table_obs_dict():
    obs = agasc.get_supplement_table("obs", as_dict=True)
    assert isinstance(obs, dict)


def test_write(tmp_path):
    import tempfile
    from pathlib import Path

    import tables

    from agasc import write_agasc

    with tables.open_file(
        Path(os.environ["SKA"]) / "data" / "agasc" / "agasc_1p7.h5"
    ) as h5_in:
        stars = Table(h5_in.root.data[:1000])

    # this is an extra column that should not make it to the output
    stars["extra_col"] = np.arange(1000)
    # channging these column types, which should then be fixed on writing
    stars["AGASC_ID"] = stars["AGASC_ID"].astype(np.int64)
    stars["MAG_ACA"] = stars["MAG_ACA"].astype(np.float64)
    stars = stars.as_array()

    temp = tmp_path / "test.h5"
    write_agasc(temp, stars=stars, version="test", order=agasc.TableOrder.DEC)
    with tables.open_file(temp) as h5_out:
        assert "data" in h5_out.root
        assert "healpix_index" not in h5_out.root
        assert h5_out.root.data.attrs["version"] == "test"
        assert h5_out.root.data.attrs["NROWS"] == 1000
        assert h5_out.root.data.dtype == agasc.TABLE_DTYPE
        assert np.all(np.diff(h5_out.root.data[:]["DEC"]) >= 0)

    write_agasc(temp, stars=stars, version="test")
    with tables.open_file(temp) as h5_out:
        assert "data" in h5_out.root
        assert "healpix_index" in h5_out.root
        assert h5_out.root.data.attrs["version"] == "test"
        assert h5_out.root.data.attrs["NROWS"] == 1000
        assert h5_out.root.data.dtype == agasc.TABLE_DTYPE
