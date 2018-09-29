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

from __future__ import print_function, division

import os
import re
from pathlib import Path

import numpy as np
import Ska.Shell
from astropy.io import ascii
from astropy.table import Table
import pytest

import agasc

# See if we can get to ASCDS environment and mp_get_agasc
ascrc_file = '{}/.ascrc'.format(os.environ['HOME'])
try:
    assert os.path.exists(ascrc_file)
    ascds_env = Ska.Shell.getenv('source {} -r release'.format(ascrc_file), shell='tcsh')
    assert 'ASCDS_BIN' in ascds_env
    cmd = 'mp_get_agasc -r 10 -d 20 -w 0.01'
    # Run the command to check for bad status (which will throw exception)
    Ska.Shell.run_shell(cmd, shell='bash', env=ascds_env)
    match = re.search(r'agasc([p0-9]+)', ascds_env['ASCDS_AGASC'])
    DS_AGASC_VERSION = match.group(1)
except Exception:
    ascds_env = None
    DS_AGASC_VERSION = None

HAS_KSH = os.path.exists('/bin/ksh')  # dependency of mp_get_agascid

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


# TO DO: update to miniagasc.h5 when DS 10.7 gets released
TEST_RADIUS = 0.6  # standard testing radius
TEST_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ['SKA'], 'data', 'agasc')
AGASC_FILE = {}
AGASC_FILE['1p6'] = DATA_DIR / 'miniagasc_1p6.h5'
AGASC_FILE['1p7'] = DATA_DIR / 'miniagasc.h5'  # Latest release

# Whether to test DS AGASC vs. agasc package HDF5 files
TEST_ASCDS = DS_AGASC_VERSION is not None and AGASC_FILE[DS_AGASC_VERSION].exists()


def random_ra_dec(nsample):
    x = np.random.uniform(-0.98, 0.98, size=nsample)
    ras = 360 * np.random.random(nsample)
    decs = np.degrees(np.arcsin(x))
    return ras, decs


def get_ds_agasc_cone(ra, dec):
    cmd = 'mp_get_agasc -r {!r} -d {!r} -w {!r}'.format(ra, dec, TEST_RADIUS)
    lines = Ska.Shell.tcsh(cmd, env=ascds_env)
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    ok1 = agasc.sphere_dist(ra, dec, dat['RA'], dat['DEC']) <= TEST_RADIUS
    ok2 = dat['MAG_ACA'] - 3.0 * dat['MAG_ACA_ERR'] / 100.0 < 11.5
    dat = dat[ok1 & ok2]

    if os.environ.get('WRITE_AGASC_TEST_FILES'):
        version = DS_AGASC_VERSION
        test_file = get_test_file(ra, dec, version)
        print(f'\nWriting {test_file} based on mp_get_agasc\n')
        dat.write(test_file, format='fits')

    return dat


def get_test_file(ra, dec, version):
    return TEST_DIR / 'data' / f'ref_ra_{ra}_dec_{dec}_{version}.fits.gz'


def get_reference_agasc_values(ra, dec, version='1p7'):
    dat = Table.read(get_test_file(ra, dec, version))
    return dat


ras = np.hstack([0., 180., 0.1, 180., 275.36])
decs = np.hstack([89.9, -89.9, 0.0, 0.0, 8.09])
# The (275.36, 8.09) coordinate fails unless date=2000:001 due to
# mp_get_agasc not accounting for proper motion.


@pytest.mark.parametrize("version", ['1p6', '1p7'])
@pytest.mark.parametrize("ra,dec", list(zip(ras, decs)))
def test_agasc_conesearch(ra, dec, version):
    """
    Compare results of get_agasc_cone to package reference data stored in
    FITS files.
    """
    try:
        ref_stars = get_reference_agasc_values(ra, dec, version=version)
    except FileNotFoundError:
        if os.environ.get('WRITE_AGASC_TEST_FILES'):
            ref_stars = agasc.get_agasc_cone(ra, dec, radius=TEST_RADIUS,
                                             agasc_file=AGASC_FILE[version],
                                             date='2000:001')
            test_file = get_test_file(ra, dec, version)
            print(f'\nWriting {test_file} based on miniagasc\n')
            ref_stars.write(test_file, format='fits')
        pytest.skip('Reference data unavailable')

    _test_agasc(ra, dec, ref_stars, version)


@pytest.mark.skipif('ascds_env is None')
@pytest.mark.parametrize("ra,dec", list(zip(ras, decs)))
def test_against_ds_agasc(ra, dec):
    """
    Compare results of get_agasc_cone to the same star field retrieved from
    the DS command line tool mp_get_agasc.
    """
    ref_stars = get_ds_agasc_cone(ra, dec)
    _test_agasc(ra, dec, ref_stars, version=DS_AGASC_VERSION)


def _test_agasc(ra, dec, ref_stars, version='1p7'):
    stars1 = agasc.get_agasc_cone(ra, dec, radius=TEST_RADIUS,
                                  agasc_file=AGASC_FILE[version],
                                  date='2000:001')
    stars1.sort('AGASC_ID')

    stars2 = ref_stars.copy()
    stars2.sort('AGASC_ID')

    # First make sure that the common stars are identical
    agasc_ids = set(stars1['AGASC_ID']).intersection(set(stars2['AGASC_ID']))
    for agasc_id in agasc_ids:
        star1 = stars1[np.searchsorted(stars1['AGASC_ID'], agasc_id)]
        star2 = stars2[np.searchsorted(stars2['AGASC_ID'], agasc_id)]
        for colname in AGASC_COLNAMES:
            if star1[colname].dtype.kind == 'f':
                assert np.all(np.abs(star1[colname] - star2[colname]) < 1e-4)
            else:
                assert star1[colname] == star2[colname]

    # Second make sure that the non-common stars are all just at the edge
    # of the faint mag limit, due to precision loss in mp_get_agasc
    for s1, s2 in ((stars1, stars2),
                   (stars2, stars1)):
        mm1 = set(s1['AGASC_ID']) - set(s2['AGASC_ID'])
        for agasc_id in mm1:
            idx = np.flatnonzero(s1['AGASC_ID'] == agasc_id)[0]
            star = s1[idx]
            bad_is_star1 = s1 is stars1
            rad = agasc.sphere_dist(ra, dec, star['RA'], star['DEC'])
            adj_mag = star['MAG_ACA'] - 3.0 * star['MAG_ACA_ERR'] / 100.0
            if adj_mag < 11.5 * 0.99:
                # Allow for loss of precision in output of mp_get_agasc
                print('Bad star', agasc_id, rad, adj_mag, bad_is_star1)
                assert False


def test_basic():
    star = agasc.get_star(1180612288)  # High-PM star
    assert np.isclose(star['RA'], 219.9053773)
    assert np.isclose(star['DEC'], -60.8371572)

    stars = agasc.get_agasc_cone(star['RA'], star['DEC'], 0.5)
    stars.sort('MAG_ACA')
    agasc_ids = [1180612176, 1180612296, 1180612184, 1180612288, 1180612192]
    mags = [-0.663, -0.576, -0.373, 0.53, 0.667]
    assert np.allclose(stars['AGASC_ID'][:5], agasc_ids)
    assert np.allclose(stars['MAG_ACA'][:5], mags)


def test_float16():
    stars = agasc.get_agasc_cone(np.float16(219.90279), np.float16(-60.83358), .015)
    assert stars['AGASC_ID'][0] == 1180612176


def test_proper_motion():
    """
    Test that the filtering in get_agasc_cone correctly expands the initial
    search radius and then does final filtering using PM-corrected positions.
    """
    star = agasc.get_star(1180612288, date='2017:001')  # High-PM star
    radius = 2.0 / 3600  # 5 arcsec

    stars = agasc.get_agasc_cone(star['RA'], star['DEC'], radius, date='2000:001')
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(star['RA'], star['DEC'], radius, date='2017:001')
    assert len(stars) == 0

    stars = agasc.get_agasc_cone(star['RA'], star['DEC'], radius, date='2017:001',
                                 pm_filter=False)
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(star['RA_PMCORR'], star['DEC_PMCORR'], radius, date='2017:001')
    assert len(stars) == 1

    stars = agasc.get_agasc_cone(star['RA_PMCORR'], star['DEC_PMCORR'], radius, date='2017:001',
                                 pm_filter=False)
    assert len(stars) == 0


@pytest.mark.parametrize(
    "agasc_id,date,ra_pmcorr,dec_pmcorr,label",
    [(1180612288, '2020:001', 219.86433151831795, -60.831868007221289, "high proper motion, epoch 2000"),
     (198451217, '2020:001', 247.89220668106938, 19.276605555555559, "epoch 1982 star"),
     (501219465, '2020:001', 166.99897608782592, 52.82208000152103, "epoch 1984 star")])
def test_add_pmcorr_is_consistent(agasc_id, date, ra_pmcorr, dec_pmcorr, label):
    """
    Check that the proper-motion corrected position is consistent reference/regress values.
    """
    star = agasc.get_star(agasc_id, date=date)
    assert np.isclose(star['RA_PMCORR'], ra_pmcorr, rtol=0, atol=1e-5)
    assert np.isclose(star['DEC_PMCORR'], dec_pmcorr, rtol=0, atol=1e-5)


def mp_get_agascid(agasc_id):
    cmd = 'mp_get_agascid {!r}'.format(agasc_id)
    lines = Ska.Shell.tcsh(cmd, env=ascds_env)
    lines = [line for line in lines if re.match(r'^\s*\d', line)]
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    return dat


@pytest.mark.skipif('not HAS_KSH')
@pytest.mark.skipif('ascds_env is None')
def test_agasc_id(radius=0.2, npointings=2, nstar_limit=5, agasc_file=AGASC_FILE):
    ras, decs = random_ra_dec(npointings)

    for ra, dec in zip(ras, decs):
        print('ra, dec =', ra, dec)
        cone_stars = agasc.get_agasc_cone(ra, dec, radius=radius, agasc_file=agasc_file)

        if len(cone_stars) == 0:
            return

        cone_stars.sort('AGASC_ID')
        for agasc_id in cone_stars['AGASC_ID'][:nstar_limit]:
            print('  agasc_id =', agasc_id)
            star1 = agasc.get_star(agasc_id, agasc_file=agasc_file)
            star2 = mp_get_agascid(agasc_id)
            for colname in AGASC_COLNAMES:
                if star1[colname].dtype.kind == 'f':
                    assert np.all(np.allclose(star1[colname], star2[colname]))
                else:
                    assert star1[colname] == star2[colname]
