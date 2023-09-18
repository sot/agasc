# coding: utf-8

"""
Create a file near_neighbor_ids.fits.gz which contains a single column
of AGASC IDs corresponding to all stars in AGASC 1.7 that are within
60 arcsec of a candidate guide or acq star.

This takes a while to run and should be done on a computer with copy
of AGASC 1.7 on a local (fast) drive.

Usage::

  $ python make_near_neighbor_ids.py --version=1p7
"""
import argparse
import os
from pathlib import Path

import tables
import tqdm
from astropy.table import Table

import agasc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version (e.g. 1.7 or 1.8rc3)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    version = args.version

    agasc_full = str(Path(os.environ["SKA"], "data", "agasc", f"agasc{version}.h5"))

    with tables.open_file(agasc_full) as h5:
        stars = h5.root.data[:]

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
    for sp in tqdm.tqdm(nears):
        near = agasc.get_agasc_cone(
            sp["RA"],
            sp["DEC"],
            radius=radius,
            date="2024:001",
            agasc_file=agasc_full,
            use_supplement=False,
            cache=True,
        )
        for id in near["AGASC_ID"]:
            if id != sp["AGASC_ID"]:
                near_ids.add(id)

    t = Table([list(near_ids)], names=["near_id"])
    t.write(f"near_neighbor_ids_{version}.fits.gz", format="fits", overwrite=True)


if __name__ == "__main__":
    main()
