# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tables
from ska_path import ska_path
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


