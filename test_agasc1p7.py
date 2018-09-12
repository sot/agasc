
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
import tables
import tables3_api
from glob import glob
import os
from astropy.io import fits
import difflib
import re


difflines = """!               THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.6
! the AXAF Guide and Acquisition Star Catalog (AGASC) version 1.6
! V1.6 changes values for MAG_ACA and MAG_ACA_ERR only.  See
! http://asc.harvard.edu/mta/ASPECT/agasc1p6cal for details
!               THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.7
! the AXAF Guide and Acquisition Star Catalog (AGASC) version 1.7
! V1.7 changes values for MAG_ACA, MAG_ACA_ERR, RSV1, RSV2 and RSV3""".split('\n')

h5_fits = tables.openFile('agasc1p7_from_fits.h5', mode='r')
h5_jupyter = tables.openFile('agasc1p7_from_jupyter.h5', mode='r')
h5_1p6 = tables.openFile('/proj/sot/ska/data/agasc/agasc1p6.h5', mode='r')

assert np.all(h5_fits.root.data.colnames == h5_1p6.root.data.colnames)
assert np.all(h5_fits.root.data.colnames == h5_jupyter.root.data.colnames)

colnames = h5_fits.root.data.colnames

ok = h5_fits.root.data.col('RSV3') == 0  # stars with no changes

agasc_ids_fits = h5_fits.root.data.col('AGASC_ID')
agasc_ids_jupyter = h5_jupyter.root.data.col('AGASC_ID')
agasc_ids_1p6 = h5_1p6.root.data.col('AGASC_ID')

idx = np.arange(len(agasc_ids_fits))

id_map_jupyter = dict(zip(agasc_ids_jupyter, idx))
id_map_1p6 = dict(zip(agasc_ids_1p6, idx))

h5_jupyter_id_map = [id_map_jupyter[agasc_id] for agasc_id in agasc_ids_fits]
h5_1p6_id_map = [id_map_1p6[agasc_id] for agasc_id in agasc_ids_fits]

for col in colnames:
    print "Processing column %s" % col

    c_fits = h5_fits.root.data.col(col)

    c_jupyter = h5_jupyter.root.data.col(col)
    c_jupyter = c_jupyter[h5_jupyter_id_map]

    c_1p6 = h5_1p6.root.data.col(col)
    c_1p6 = c_1p6[h5_1p6_id_map]

    # All values for all columns are identical between the V1.7 h5 files
    # from fits files and from jupyter notebook
    assert np.all(c_fits == c_jupyter)

    if col not in ['RSV1', 'RSV2', 'RSV3', 'MAG_ACA', 'MAG_ACA_ERR']:
        # All values for all columns except MAG_ACA, MAG_ACA_ERR, RSV1-3
        # are identical between V1.7 and V1.6
        assert np.all(c_fits == c_1p6)
    else:
        # We see only expected differences in MAG_ACA, MAG_ACA_ERR, RSV1-3
        # between V1.7 and V1.6
        assert np.all(c_fits[ok] == c_1p6[ok])
        assert np.all(c_fits[~ok] != c_1p6[~ok])

h5_fits.close()
h5_jupyter.close()
h5_1p6.close()

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
