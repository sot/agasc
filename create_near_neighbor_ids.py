# coding: utf-8

"""
Create a file near_neighbor_ids.fits.gz which contains a single column
of AGASC IDs corresponding to all stars in AGASC 1.7 that are within
60 arcsec of a candidate guide or acq star.

This takes a while to run and should be done on a computer with copy
of AGASC 1.7 on a local (fast) drive.

Usage::

  $ python make_near_neighbor_ids.py
"""
import tables
import os
from pathlib import Path

import agasc
from astropy.table import Table

agasc1p7 = str(Path(os.environ["SKA"], "data", "agasc", "agasc1p7.h5"))

h5 = tables.open_file(agasc1p7)
stars = h5.root.data[:]
h5.close()

ok = (
    (stars["CLASS"] == 0)
    & (stars["MAG_ACA"] < 11.0)
    & (stars["ASPQ1"] < 50)
    & (stars["ASPQ1"] > 0)  # Less than 2.5 arcsec from nearby star
    & (stars["ASPQ2"] == 0)  # Proper motion less than 0.5 arcsec/yr
)

# Candidate acq/guide stars with a near neighbor that made ASPQ1 > 0
nears = stars[ok]

radius = 60 / 3600
near_ids = set()
for ii, sp in enumerate(nears):
    near = agasc.get_agasc_cone(
        sp["RA"], sp["DEC"], radius=radius, date="2000:001", agasc_file=agasc1p7
    )
    for id in near["AGASC_ID"]:
        if id != sp["AGASC_ID"]:
            near_ids.add(id)

    if ii % 100 == 0:
        print(ii)

t = Table([list(near_ids)], names=["near_id"])
t.write("near_neighbor_ids_1p7.fits.gz", format="fits")
