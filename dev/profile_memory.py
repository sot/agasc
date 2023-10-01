import os
from pathlib import Path

import proseco
import sparkles

import agasc

STD_INFO = dict(
    att=(0, 0, 0),
    detector="ACIS-S",
    sim_offset=0,
    focus_offset=0,
    date="2018:001",
    n_guide=5,
    n_fid=3,
    t_ccd=-11,
    man_angle=90,
    dither=8.0,
)


print(f"agasc.__version__ = {agasc.__version__}")
print(f"proseco.__version__ = {proseco.__version__}")
print(f"sparkles.__version__ = {sparkles.__version__}")

# This does nothing in ska3-flight <= 2023.3.
os.environ["AGASC_HDF5_FILE"] = str(
    Path.home() / "git" / "agasc" / "proseco_agasc_healpix_1p7.h5"
)

aca = proseco.get_aca_catalog(**STD_INFO)
acar = aca.get_review_table()
acar.run_aca_review()
print(acar.messages)
