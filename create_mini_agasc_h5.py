import tables
import Ska.Table


example_file = '/data/agasc1p6/agasc/n0000/0001.fit'
dtype = Ska.Table.read_fits_table(example_file).dtype
table_desc, bo = tables.descr_from_dtype(dtype)
#filters = tables.Filters(complevel=5, complib='zlib')
minih5 = tables.openFile('miniagasc.h5', mode='w')
minitbl = minih5.createTable('/', 'data', table_desc,
                             title='AGASC 1.6')
minitbl.flush()
minih5.flush()

h5 = tables.openFile('agasc1p6.h5', mode='r')
tbl = h5.getNode("/", 'data')
ok = tbl.getWhereList("(MAG_ACA - (3 * MAG_ACA_ERR / 100)) < 11.5")
stars = tbl.readCoordinates(ok)

minitbl.append(stars)
minitbl.flush()

minitbl.cols.RA.createCSIndex()
minitbl.cols.DEC.createCSIndex()
minitbl.cols.AGASC_ID.createCSIndex()
minitbl.flush()

minih5.flush()
minih5.close()

h5.close()
