import numpy as np
import tables
import Ska.Table
from astropy.table import Table


example_file = '/data/agasc1p6/agasc/n0000/0001.fit'
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)

print 'Reading full agasc and selecting useable stars'
h5 = tables.openFile('agasc1p6.h5', mode='r')
tbl = h5.getNode("/", 'data')
idxs = tbl.getWhereList("(MAG_ACA - (3.0 * MAG_ACA_ERR / 100.0)) < 11.5")
stars = tbl.readCoordinates(idxs)
h5.close()

print 'Sorting on Dec and re-ordering'
idx = np.argsort(stars['DEC'])
stars = stars.take(idx)

ra = np.array(stars['RA'], dtype='f8')
dec = np.array(stars['DEC'], dtype='f8')

print 'Saving lightweight index ra_dec.npy'
out = Table([ra, dec], names=('ra', 'dec'))
np.save('ra_dec.npy', out)

print 'Creating miniagasc.h5 file'
minih5 = tables.openFile('miniagasc.h5', mode='w')
minitbl = minih5.createTable('/', 'data', table_desc,
                             title='AGASC 1.6')
print 'Appending stars to miniagasc.h5 file'
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
