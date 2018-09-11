
import numpy as np
import tables

h5 = tables.openFile('agasc1p7.h5', mode='r')
h5inp = tables.openFile('/proj/sot/ska/data/agasc/agasc1p7.h5', mode='r')
h51p6 = tables.openFile('/proj/sot/ska/data/agasc/agasc1p6.h5', mode='r')

assert np.all(h5.root.data.colnames == h51p6.root.data.colnames)
assert np.all(h5.root.data.colnames == h5inp.root.data.colnames)

colnames = h5.root.data.colnames

# stars with no changes
ok = h5.root.data.col('RSV3') == 0

agasc_ids = h5.root.data.col('AGASC_ID')
agasc_ids2 = h5inp.root.data.col('AGASC_ID')
agasc_ids3 = h51p6.root.data.col('AGASC_ID')

idx = np.arange(len(agasc_ids))

id_map2 = dict(zip(agasc_ids2, idx))
id_map3 = dict(zip(agasc_ids3, idx))

h5inp_id_map = [id_map2[agasc_id] for agasc_id in agasc_ids]
h51p6_id_map = [id_map3[agasc_id] for agasc_id in agasc_ids]

for col in colnames:
    c1 = h5.root.data.col(col)

    c2 = h5inp.root.data.col(col)
    c2 = c2[h5inp_id_map]

    c3 = h51p6.root.data.col(col)
    c3 = c3[h51p6_id_map]

    assert np.all(c1 == c2)

    if col not in ['RSV1', 'RSV2', 'RSV3', 'MAG_ACA', 'MAG_ACA_ERR']:
        assert np.all(c1 == c3)
    else:
        assert np.all(c1[ok] == c3[ok])

h5.close()
h5inp.close()
h51p6.close()
