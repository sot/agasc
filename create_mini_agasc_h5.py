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
parser.add_argument('--ignore-near-neighbors',
                    action='store_true',
                    help='Ignore near neighbor stars')
parser.add_argument('--include-rarely-used-cols',
                    action='store_true',
                    help='Include columns that are rarely used in operations')
args = parser.parse_args()

num_version = re.sub(r'p', '.', args.version)

filename = 'agasc{}.h5'.format(args.version)
print 'Reading full AGASC {} and selecting useable stars'.format(filename)

h5 = tables.openFile(filename)
stars = h5.root.data[:]
h5.close()

# Filter mags
ok = stars['MAG_ACA'] - 3.0 * stars['MAG_ACA_ERR'] / 100.0 < 11.5

# Put back near-neighbor stars that got cut by above mag filter. This file
# is made with create_near_neighbor_ids.py.
if not args.ignore_near_neighbors:
    near_file = 'near_neighbor_ids_{}.fits.gz'.format(args.version)
    near_table = Table.read(near_file, format='fits')
    near_ids = set(near_table['near_id'])
    print "Including {} near neighbor stars".format(len(near_ids))
    for idx, agasc_id in enumerate(stars['AGASC_ID']):
        if agasc_id in near_ids:
            ok[idx] = True

# Filter down to miniagasc stars
print 'Filtering from {} to {} stars'.format(len(stars), np.count_nonzero(ok))
stars = stars[ok]

if not args.include_rarely_used_cols:
    print 'Excluding rarely used columns'
    excludes = ['PLX', 'PLX_ERR', 'PLX_CATID',
                'ACQQ1', 'ACQQ2', 'ACQQ3', 'ACQQ4', 'ACQQ5', 'ACQQ6',
                'XREF_ID1', 'XREF_ID2', 'XREF_ID3', 'XREF_ID4', 'XREF_ID5',
                'RSV4', 'RSV5', 'RSV6']
    names = [name for name in stars.dtype.names if name not in excludes]
    print 'Dtype before excluding:\n', stars.dtype
    stars = Table([stars[name] for name in names], names=names, copy=False)
    stars = stars.as_array()
    print 'Dtype after excluding:\n', stars.dtype

print 'Sorting on Dec and re-ordering'
idx = np.argsort(stars['DEC'])
stars = stars.take(idx)

print 'Creating miniagasc.h5 file'
filename = 'miniagasc_{}.h5'.format(args.version)
table_desc, bo = tables.descr_from_dtype(stars.dtype)
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
