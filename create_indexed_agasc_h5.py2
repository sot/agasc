# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import Ska.Table
import tables

example_file = '/proj/sot/ska/data/agasc1p7/agasc/n0000/0001.fit'
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)
h5 = tables.openFile('agasc1p7_from_fits.h5', mode='r')
tbl = h5.root.data[:]

print 'Sorting on Dec and re-ordering'
idx = np.argsort(tbl['DEC'])

h5_indexed = tables.openFile('agasc1p7_from_fits_indexed.h5', mode='w')
tbl_indexed = h5_indexed.createTable('/', 'data', table_desc,
                     title='AGASC 1.7')
tbl_indexed.append(tbl[idx])
tbl_indexed.flush()

print 'Creating indexes in agasc1p7_from_fits_indexed.h5 file'
tbl_indexed.cols.RA.createCSIndex()
tbl_indexed.cols.DEC.createCSIndex()
tbl_indexed.cols.AGASC_ID.createCSIndex()

tbl_indexed.flush()
h5_indexed.flush()
h5_indexed.close()

h5.flush()
h5.close()
