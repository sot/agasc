# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tables
from ska_path import ska_path
import numpy as np
from astropy.table import Table
import tempfile

from .. import agasc

DATA_ROOT = ska_path('data', 'agasc')
tables_open_file = getattr(tables, 'open_file', None) or tables.openFile

def test_multi_agasc():

    tempdir = tempfile.mkdtemp()

    # Make two custom agasc files from the miniagasc, using 20 stars from
    # around the middle of the table
    with tables_open_file(os.path.join(DATA_ROOT, 'miniagasc.h5')) as h5:
        middle = int(len(h5.root.data) // 2)
        stars1 = Table(h5.root.data[middle:middle+20])
        stars1.write(os.path.join(tempdir, 'stars1.h5'), path='data')
        stars2 = Table(h5.root.data[middle + 20:middle + 60])
        stars2.write(os.path.join(tempdir, 'stars2.h5'), path='data')

    # Fetch all the stars from a custom agasc and make sure we have the right number of stars
    # with no errors
    all_stars2 = agasc.get_agasc_cone(0, 90, radius=180, agasc_file=os.path.join(tempdir, 'stars2.h5'))
    assert len(all_stars2) == len(stars2)
    # Fetch all the stars from the other custom agasc and do the same.  The point of the two files
    # is to confirm that the caching behavior in agasc doesn't cause problems with fetches
    all_stars1 = agasc.get_agasc_cone(0, 90, radius=180, agasc_file=os.path.join(tempdir, 'stars1.h5'))
    assert len(all_stars1) == len(stars1)

    # Do a position filtered search using the first star in the table as a reference and make sure
    # we get the same star from the reference agasc.  Do this with the stars2 file as this confirms
    # that we can switch back and forth between files and get the correct content.
    cone2 = agasc.get_agasc_cone(all_stars2['RA'][0], all_stars2['DEC'][0], radius=0.000001,
                                 agasc_file=os.path.join(tempdir, 'stars2.h5'))
    # And this is a read of the default agasc file after the custom ones so should confirm that
    # the custom files didn't break that access.
    cone2_full = agasc.get_agasc_cone(all_stars2['RA'][0], all_stars2['DEC'][0], radius=0.000001)
    assert cone2[0]['AGASC_ID'] == cone2_full[0]['AGASC_ID']
    # Confirm that there is just one star in this test setup (not a module test, but confirms test
    # setup is as intended).
    assert len(cone2_full) == 1
    assert len(cone2) == len(cone2_full)


def test_update_color1_func():
    """
    Test code to update the COLOR1 column.
    """
    color1 = [1.0, 1.0, 1.5, 1.5, 1.5, 1.5]
    color2 = np.array([1.0, 1.0, 1.5, 1.5, 1.75, 2.0]) / 0.850
    rsv3 = [0, 1, 0, 1, 0, 1]
    # First two are left unchanged because color1 < 1.5
    # Third is still 1.5 because RSV3=0 (no good mag available so still "bad mag")
    # Fourth now gets COLOR1 = 1.499 because RSV3=1 => good mag
    # Fifth is still 1.5 because RSV3=0 (no good mag available so still "bad mag")
    # Sixth now gets COLOR1 = COLOR2 * 0.850 = 2.0
    stars = Table([color1, color2, rsv3], names=['COLOR1', 'COLOR2', 'RSV3'])
    agasc.update_color1_column(stars)

    assert np.allclose(stars['COLOR1'], [1.0, 1.0, 1.5, 1.499, 1.5, 2.0])
    assert np.allclose(stars['COLOR2'], color2)


def test_update_color1_get_star():
    """
    Test updated color1 in get_star() call.

     AGASC_ID  COLOR1  COLOR2 C1_CATID  RSV3
      int32   float32 float32  uint8   uint8
    --------- ------- ------- -------- -----
    981997696     1.5   1.765        5     0
    759439648     1.5   1.765        5     1

    """
    star = agasc.get_star(981997696)
    assert np.isclose(star['COLOR1'], 1.5)

    star = agasc.get_star(759439648)
    assert np.isclose(star['COLOR1'], 1.499)

    star = agasc.get_star(981997696, fix_color1=False)
    assert np.isclose(star['COLOR1'], 1.5)

    star = agasc.get_star(759439648, fix_color1=False)
    assert np.isclose(star['COLOR1'], 1.5)


def test_update_color1_get_agasc_cone():
    """
    Test updated color1 in get_agasc_cone() call.
    """
    ra, dec = 323.22831196, -13.11621348
    stars = agasc.get_agasc_cone(ra, dec, 0.2)
    stars.add_index('AGASC_ID')
    assert np.isclose(stars.loc[759960152]['COLOR1'], 1.60055, rtol=0, atol=0.0005)
    assert np.isclose(stars.loc[759439648]['COLOR1'], 1.499, rtol=0, atol=0.0005)

    stars = agasc.get_agasc_cone(ra, dec, 0.2, fix_color1=False)
    stars.add_index('AGASC_ID')
    assert np.isclose(stars.loc[759960152]['COLOR1'], 1.5, rtol=0, atol=0.0005)
    assert np.isclose(stars.loc[759439648]['COLOR1'], 1.5, rtol=0, atol=0.0005)
