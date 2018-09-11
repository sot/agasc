# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import Ska.Table
import tables
import os
from glob import glob
import Ska.Numpy


example_file = '/proj/sot/ska/data/agasc1p7/agasc/n0000/0001.fit'
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)
#filters = tables.Filters(complevel=5, complib='zlib')
h5 = tables.openFile('agasc1p7.h5', mode='w')
tbl = h5.createTable('/', 'data', table_desc,
                     title='AGASC 1.7')
tbl.flush()
h5.flush()
h5.close()

AGASC_DIR = '/proj/sot/ska/data/agasc1p7/agasc'
h5 = tables.openFile('agasc1p7.h5', mode='a')
tbl = h5.getNode('/', 'data')
for chunk in glob(os.path.join(AGASC_DIR, "?????")):
    for file in glob(os.path.join(chunk, "????.fit")):
        print "processing %s" % file
        stars = Ska.Table.read_table(file)
        tbl.append(stars)
        tbl.flush()
        h5.flush()

print 'Creating indexes in agasc1p7.h5 file'
tbl.cols.RA.createCSIndex()
tbl.cols.DEC.createCSIndex()
tbl.cols.AGASC_ID.createCSIndex()

h5.flush()
h5.close()
