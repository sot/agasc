
"""
Test script for the AGASC V1.7.

Files required for the test:
    1. Input h5 file (agasc1p7_from_jupyter.h5) created from jupyter notebook,
       agasc_apass_recalibration.ipynb in /proj/sot/ska/www/ASPECT/ipynb
    2. AGASC V1.6 and V1.7 fits files.
    3. New h5 file (agasc1p7_from_fits.h5) created from fits files using the
       create_agasc_h5.py script.
    4. AGASC V1.6 h5 file.

The test consists of the following steps:
    1. Confirm that all the fields in all the columns in agasc1p7_from_fits.h5
       file match values in the input agasc1p7_jupter.h5 file.
    2. Confirm that all the fields in all columns other than MAG_ACA,
       MAG_ACA_ERR, RSV1-3 in agasc1p7_from_fits.h5 file match values
       in V1.6 h5 file.
    3. Confirm that we see expected differences between agasc1p7_from_fits.h5
       file relative to the V1.6 h5 file (MAG_ACA, MAG_ACA_ERR, RSV1-3 cols)
    4. Verify that comment changes in the fits files are as expected, diff
       of the comments cards for all fits files.
"""
import numpy as np
from glob import glob
import os
from astropy.io import fits
from astropy.table import Table
import difflib
import re


difflines = """!               THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.6
! the AXAF Guide and Acquisition Star Catalog (AGASC) version 1.6
! V1.6 changes values for MAG_ACA and MAG_ACA_ERR only.  See
! http://asc.harvard.edu/mta/ASPECT/agasc1p6cal for details
!               THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.7
! the AXAF Guide and Acquisition Star Catalog (AGASC) version 1.7
! V1.7 changes values for MAG_ACA, MAG_ACA_ERR, RSV1, RSV2 and RSV3""".split('\n')

h5_fits = Table.read('agasc1p7_from_fits.h5', path='data')
h5_jupyter = Table.read('agasc1p7_from_jupyter.h5', path='data')
h5_1p6 = Table.read('/proj/sot/ska/data/agasc/agasc1p6.h5', path='data')

# Sort by AGASC_ID, and RA to account for the case of duplicate 154534513 star
h5_fits.sort(['AGASC_ID', 'RA'])
h5_jupyter.sort(['AGASC_ID', 'RA'])
h5_1p6.sort(['AGASC_ID', 'RA'])

assert np.all(h5_fits.dtype.names == h5_1p6.dtype.names)
assert np.all(h5_fits.dtype.names == h5_jupyter.dtype.names)

ok = h5_fits['RSV3'] == 0  # stars with no changes

for col in h5_fits.dtype.names:
    print "Processing column %s" % col

    c_fits = h5_fits[col]
    c_jupyter = h5_jupyter[col]
    c_1p6 = h5_1p6[col]

    # All values for all columns, except the RSV1 column, are identical between
    # the V1.7 h5 files from fits files and from jupyter notebook

    if col == 'RSV1':
        # fits have RSV1 overwritten with -9999 for stars that did not change
        assert np.all(c_fits[ok] == -9999)
        assert np.all(c_fits[~ok] != -9999)
    else:
        assert np.all(c_fits == c_jupyter)

    test_cols = ('RSV1', 'RSV2', 'RSV3', 'MAG_ACA', 'MAG_ACA_ERR')

    if col in test_cols:
        # We see only expected differences in MAG_ACA, MAG_ACA_ERR, RSV1-3
        # between V1.7 and V1.6

        assert np.all(c_jupyter[ok] == c_1p6[ok])
        if col != 'RSV1':
            assert np.all(c_fits[ok] == c_1p6[ok])

        if col != 'MAG_ACA_ERR':
            # because ~1% of stars have MAG_ACA_ERR change that is too small to
            # be reflected in the new h5 file where MAG_ACA_ERR is a small int
            assert np.all(c_fits[~ok] != c_1p6[~ok])
    else:
        # All values for all columns except MAG_ACA, MAG_ACA_ERR, RSV1-3
        # are identical between V1.7 and V1.6
        assert np.all(c_fits == c_1p6)


# Diff of the comment cards for all V1.6 and V1.7 fits files
AGASC_DIR_1p6 = '/proj/sot/ska/data/agasc1p6/agasc'
for chunk in glob(os.path.join(AGASC_DIR_1p6, "?????")):
    print "processing fits files in chunk %s" % chunk
    for file_1p6 in glob(os.path.join(chunk, "????.fit")):
        file_1p7 = file_1p6.replace('agasc1p6', 'agasc1p7')

        hdu_1p6 = fits.open(file_1p6)
        hdu_1p7 = fits.open(file_1p7)
        comment_1p6 = hdu_1p6[0].header['COMMENT']
        comment_1p7 = hdu_1p7[0].header['COMMENT']
        d = difflib.context_diff(comment_1p6, comment_1p7)
        dls = [line for line in d]
        ok = [re.search('1[.pvV][67]', item) is not None for item in dls]
        ok = np.array(ok, dtype=bool)

        assert np.all(np.array(dls)[ok] == np.array(difflines))

        hdu_1p6.close()
        hdu_1p7.close()
