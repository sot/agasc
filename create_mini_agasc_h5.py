# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import re

import numpy as np
import tables
import Ska.Table
from astropy.table import Table

parser = argparse.ArgumentParser()
parser.add_argument('--version',
                    default='1p7',
                    help='Version (e.g. 1p6 or 1p7, default=1p7')
args = parser.parse_args()

num_version = re.sub(r'p', '.', args.version)

example_file = '/proj/sot/ska/data/agasc{}/agasc/n0000/0001.fit'.format(args.version)
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)

filename = 'agasc{}.h5'.format(args.version)
print 'Reading full AGASC {} and selecting useable stars'.format(filename)
h5 = tables.openFile(filename, mode='r')
tbl = h5.getNode("/", 'data')
idxs = tbl.getWhereList("(MAG_ACA - (3.0 * MAG_ACA_ERR / 100.0)) < 11.5")
stars = tbl.readCoordinates(idxs)
h5.close()

print 'Sorting on Dec and re-ordering'
idx = np.argsort(stars['DEC'])
stars = stars.take(idx)

print 'Creating miniagasc.h5 file'
filename = 'miniagasc_{}.h5'.format(args.version)
minih5 = tables.openFile(filename, mode='w')
minitbl = minih5.createTable('/', 'data', table_desc,
                             title='AGASC {}'.format(num_version))
print 'Appending stars to {} file'.format(filename)
minitbl.append(stars)
minitbl.flush()

print 'Creating indexes in miniagasc.h5 file'
minitbl.cols.RA.createCSIndex()
minitbl.cols.DEC.createCSIndex()
minitbl.cols.AGASC_ID.createCSIndex()

print 'Flush and close miniagasc.h5 file'
minitbl.flush()
minih5.flush()
minih5.close()
