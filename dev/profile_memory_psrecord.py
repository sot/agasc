sleep_time = 2

print("importing time")
import time

t0 = time.time()


def wait_until_time(t1):
    t_now = time.time()
    time.sleep(t1 - (t_now - t0))


print(f"import other dependencies at {time.time() - t0:.2f} s")
import contextlib
import functools
import os
import re
from packaging.version import Version
from pathlib import Path
from typing import Optional

import numexpr
import numpy as np
import tables
from astropy.table import Column, Table
from Chandra.Time import DateTime
import tables

wait_until_time(2.0)

print(f"importing agasc at {time.time() - t0:.2f} s")
import agasc

print(f"agasc.__file__ = {agasc.__file__}")
print(f"agasc.__version__ = {agasc.__version__}")

wait_until_time(4.0)

if "AGASC_HDF5_FILE" in os.environ:
    agasc_file = os.environ["AGASC_HDF5_FILE"]
else:
    agasc_file = agasc.default_agasc_dir() / "proseco_agasc_1p7.h5"
print(f"getting stars from {agasc_file} at {time.time() - t0:.2f} s")
stars = agasc.get_agasc_cone(0, 0, agasc_file=agasc_file)

wait_until_time(6.0)

print(f"importing proseco at {time.time() - t0:.2f} s")
import proseco

wait_until_time(8.0)

print(f"importing sparkles at {time.time() - t0:.2f} s")
import sparkles

wait_until_time(10.0)

from proseco.tests.test_common import STD_INFO

print(f"running get_aca_catalog at {time.time() - t0:.2f} s")
aca = proseco.get_aca_catalog(**STD_INFO)

wait_until_time(12.0)

print(f"running get_aca_review at {time.time() - t0:.2f} s")
acar = aca.get_review_table()
acar.run_aca_review()

wait_until_time(14.0)
