#!/usr/bin/env python

"""
Create index files for fast access to mini-AGASC.
"""

import numpy as np
import tables

from astropy.table import Table
import Ska.Table

print 'Reading unsorted miniagasc'
h5 = tables.openFile('miniagasc_raw.h5', 'r')
dat = h5.root.data[:]
h5.close()

print 'Sorting and re-ordering'
idx = np.argsort(dat['DEC'])
dat = dat.take(idx)

ra = np.array(dat['RA'], dtype='f8')
dec = np.array(dat['DEC'], dtype='f8')

print 'Saving ra_dec.npy'
out = Table([ra, dec], names=('ra', 'dec'))
np.save('ra_dec.npy', out)

example_file = '/data/agasc1p6/agasc/n0000/0001.fit'
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)

h5 = tables.openFile('miniagasc.h5', mode='w')

print 'Writing miniagasc.h5'
tbl = h5.createTable('/', 'data', table_desc, title='AGASC 1.6')
tbl.append(dat)
tbl.flush()
tbl.cols.AGASC_ID.createCSIndex()
tbl.flush()

h5.close()
