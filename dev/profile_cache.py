"""Profile performance of get_agasc_cone with and without cache"""

import itertools
import time
from pathlib import Path

import numpy as np

from agasc import __version__, get_agasc_cone, get_agasc_filename

print(f"agasc version: {__version__}")

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


def print_timing(filename, cache):
    t0 = 0
    for cols in (None, colnames):
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
        print(
            f"filename={Path(filename).name} {cache=} columns:{cols is not None} {dt_ms:.2f} ms"
        )


print_timing(agasc_dec, cache=False)
print_timing(agasc_dec, cache=True)
print_timing(agasc_healpix, cache=False)
print_timing(agasc_healpix, cache=True)
