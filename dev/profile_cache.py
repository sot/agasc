"""Profile performance of get_agasc_cone with and without cache"""
import time
from pathlib import Path

import numpy as np

from agasc import get_agasc_cone, get_agasc_filename

agasc_healpix = get_agasc_filename("proseco_agasc_*", version="1p8", allow_rc=True)
agasc_dec = get_agasc_filename("proseco_agasc_*", version="1p7")

ras = np.random.uniform(0, 360, 100)
decs = np.random.uniform(-90, 90, 100)


def print_timing(filename, cache):
    t0 = time.time()
    for ra, dec in zip(ras, decs):
        get_agasc_cone(
            ra,
            dec,
            radius=1.5,
            agasc_file=filename,
            date="2019:001",
            cache=cache,
        )
    print(f"filename={Path(filename).name} {cache=} {time.time() - t0:.2f}")


print_timing(agasc_dec, cache=False)
print_timing(agasc_dec, cache=True)
print_timing(agasc_healpix, cache=False)
print_timing(agasc_healpix, cache=True)
