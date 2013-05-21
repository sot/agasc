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
>>> import test_agasc
>>> test_agasc.test_agasc(nsample=100)
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


def mp_get_agasc(ra, dec, radius):
    tcsh_cmd = 'source {}/.ascrc -r release; mp_get_agasc -r {!r} -d {!r} -w {!r}'.format(
        os.environ['HOME'], ra, dec, radius * 1.5)
    cmd = 'tcsh -c {!r}'.format(tcsh_cmd)
    lines = Ska.Shell.bash(cmd)
    i_start = [i for i, line in enumerate(lines) if line.startswith('CXC Data System')][-1]
    lines = lines[i_start + 1:]
    dat = ascii.read(lines, Reader=ascii.NoHeader, names=AGASC_COLNAMES)

    ok1 = agasc.sphere_dist(ra, dec, dat['RA'], dat['DEC']) <= radius
    ok2 = dat['MAG_ACA'] - 3.0 * dat['MAG_ACA_ERR'] / 100.0 < 11.5
    dat = dat[ok1 & ok2]

    return dat


def test_agasc(radius=1.4, nsample=2):
    ras, decs = random_ra_dec(nsample)
    ras = np.hstack([ras, [0., 180., 0.1, 180.]])
    decs = np.hstack([decs, [89.9, -89.9, 0.0, 0.0]])

    for ra, dec in zip(ras, decs):
        print ra, dec

        stars1 = agasc.get_agasc_cone(ra, dec, radius=radius, agasc_file='miniagasc.h5')
        stars1.sort('AGASC_ID')

        stars2 = mp_get_agasc(ra, dec, radius)
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
                    print 'Bad star', agasc_id, rad, adj_mag, bad_is_star1
                    assert False
