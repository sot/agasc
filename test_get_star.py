"""
Test that the information from agasc.get_star() matches the lookup information
from the ASCDS tool mp_get_agascid.  

This uses some random ra/dec with a cone search to get some star ids, and then 
those ids are compared.

Comprehensive test (takes a while):
>>> import test_get_star
>>> test_get_star.test_agasc_id()
"""


import os

import numpy as np
import Ska.Shell
from astropy.io import ascii

import agasc

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


def random_ra_dec(nsample):
    x = np.random.uniform(-0.98, 0.98, size=nsample)
    ras = 360 * np.random.random(nsample)
    decs = np.degrees(np.arcsin(x))
    return ras, decs


def mp_get_agascid(agasc_id):
    tcsh_cmd = 'source {}/.ascrc -r release; mp_get_agascid {!r}'.format(
        os.environ['HOME'], agasc_id)
    cmd = 'tcsh -c {!r}'.format(tcsh_cmd)
    lines = Ska.Shell.bash(cmd)
    i_start = [i for i, line in enumerate(lines) if line.startswith('AGASC_INPUT_FILEMAP')][-1]
    # start two lines after AGASC_INPUT_FILEMAP
    lines = lines[i_start + 2:]
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    return dat


def test_agasc_id(radius=.2, npointings=50, nstar_limit=50):
    ras, decs = random_ra_dec(npointings)
    ras = np.hstack([ras, [0., 180., 0.1, 180.]])
    decs = np.hstack([decs, [89.9, -89.9, 0.0, 0.0]])

    stars_checked = 0
    for ra, dec in zip(ras, decs):
        print ra, dec

        cone_stars = agasc.get_agasc_cone(ra, dec, radius=radius, agasc_file='miniagasc.h5')
        if not len(cone_stars):
            continue
        cone_stars.sort('AGASC_ID')
        for agasc_id in cone_stars['AGASC_ID']:
            print agasc_id
            star1 = agasc.get_star(agasc_id)
            star2 = mp_get_agascid(agasc_id)
            for colname in AGASC_COLNAMES:
                if star1[colname].dtype.kind == 'f':
                    assert np.all(np.abs(star1[colname] - star2[colname]) < 1e-4)
                else:
                    assert star1[colname] == star2[colname]
            stars_checked += 1
            print stars_checked
        if stars_checked > nstar_limit:
            break
