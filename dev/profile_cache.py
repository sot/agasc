"""Profile performance of get_agasc_cone with and without cache"""

import itertools
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from agasc import __version__, get_agasc_cone, get_agasc_filename

agasc_healpix = get_agasc_filename("proseco_agasc_*", version="1p8", allow_rc=True)
agasc_dec = get_agasc_filename("proseco_agasc_*", version="1p7")

n_iter = 300
np.random.seed(0)
ras = np.random.uniform(0, 360, n_iter)
decs = np.random.uniform(-90, 90, n_iter)

colnames = (
    "AGASC_ID",
    "RA",
    "DEC",
    "PM_RA",
    "PM_DEC",
    "EPOCH",
    "MAG_ACA",
)


def get_timing(filename, cache, cols):
    t0 = 0
    for idx, ra, dec in zip(itertools.count(), ras, decs):
        get_agasc_cone(
            ra,
            dec,
            radius=1.5,
            agasc_file=filename,
            date="2019:001",
            cache=cache,
            columns=cols,
            fix_color1=False,
            use_supplement=False,
            pm_filter=False,
        )
        if idx == 0:
            t0 = time.time()
    dt_ms = (time.time() - t0) / (len(ras) - 1) * 1000
    row = (Path(filename).name, cache, cols is None, dt_ms)
    return row


rows = []
for filename in (agasc_dec, agasc_healpix):
    for cache in (False, True):
        for cols in (None, colnames):
            print(".", end="", flush=True)
            rows.append(get_timing(filename, cache, cols))

out = Table(rows=rows, names=("filename", "cache", "all_columns", "dt"))
out["dt"].format = ".2f"
out["dt"].unit = "ms"

print()
print(f"agasc version: {__version__}")
out.pprint_all()
