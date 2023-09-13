# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
create_mini_agasc_h5
====================

This script creates a mini AGASC HDF5 file from the full AGASC catalog.

The output can be either ``miniagasc_<version>.h5`` or ``proseco_agasc_<version>.h5``.

Examples
--------

Most common usage::

  $ python create_mini_agasc_h5.py --version 1.8 --include-near-neighbors
  $ python create_mini_agasc_h5.py --version 1.8 --proseco
"""
import argparse

import numpy as np
import tables
from astropy.table import Table

from agasc.healpix import get_healpix


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version (e.g. 1.7 or 1.8rc3)",
    )
    parser.add_argument(
        "--include-near-neighbors",
        action="store_true",
        help="Include near neighbor stars even if filtered out by magnitude",
    )
    parser.add_argument(
        "--proseco",
        action="store_true",
        help="Create proseco_agasc (this implies --include-near-neighbors)",
    )
    parser.add_argument(
        "--healpix",
        action="store_true",
        help="Create AGASC file using HEALpix ordering with a healpix_index table",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    version_num = args.version
    version = version_num.replace(".", "p")
    rootname = "proseco_agasc" if args.proseco else "miniagasc"

    filename_full = f"agasc{version}.h5"
    filename_mini = f"{rootname}_{args.version}.h5"

    stars = get_mini_agasc_stars(
        filename_full,
        version,
        include_near_neighbors=args.include_near_neighbors or args.proseco,
    )

    if args.proseco:
        stars = filter_proseco_stars(stars)

    if args.healpix:
        print("Creating healpix_index table and sorting by healpix index")
        healpix_index, idx_sort = get_healpix_index_table(stars)
    else:
        print("Sorting on DEC column")
        idx_sort = np.argsort(stars["DEC"])
    stars = stars.take(idx_sort)

    write_mini_agasc(args, version_num, stars)
    if args.healpix:
        write_healpix_index_table(filename_mini, healpix_index)


def write_healpix_index_table(filename: str, healpix_index: Table, nside: int):
    """
    Write a HEALPix index table to an HDF5 file.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to write to.
    healpix_index : astropy.table.Table
        The HEALPix index table to write.
    nside : int
        The NSIDE parameter used to generate the HEALPix index.

    Returns
    -------
    None
    """
    healpix_index_np = healpix_index.as_array()

    with tables.open_file(filename, mode="a") as h5:
        h5.create_table("/", "healpix_index", healpix_index_np, title="HEALPix index")
        h5.root.healpix_index.attrs["nside"] = nside


def write_mini_agasc(filename: str, stars: np.ndarray, version_num: str):
    print(f"Creating {filename}")

    with tables.open_file(filename, mode="w") as h5:
        data = h5.create_table("/", "data", stars, title=f"AGASC {version_num}")
        data.attrs["version"] = version_num
        data.flush()

        print("  Creating AGASC_ID index")
        data.cols.AGASC_ID.create_csindex()

        print(f"  Flush and close {filename}")
        data.flush()
        h5.flush()


def filter_proseco_stars(stars):
    print("Excluding columns not needed for proseco")
    # fmt: off
    excludes = ['PLX', 'PLX_ERR', 'PLX_CATID',
                'ACQQ1', 'ACQQ2', 'ACQQ3', 'ACQQ4', 'ACQQ5', 'ACQQ6',
                'XREF_ID1', 'XREF_ID2', 'XREF_ID3', 'XREF_ID4', 'XREF_ID5',
                'RSV4', 'RSV5', 'RSV6',
                'POS_CATID', 'PM_CATID',
                'MAG', 'MAG_ERR', 'MAG_BAND', 'MAG_CATID',
                'COLOR1_ERR', 'C1_CATID',  # Keep color1, 2, 3
                'COLOR2_ERR', 'C2_CATID',
                'RSV2',
                'VAR_CATID']
    # fmt: on

    names = [name for name in stars.dtype.names if name not in excludes]
    print("Dtype before excluding:\n", stars.dtype)
    stars = Table({name: stars[name] for name in names}, copy=False)
    stars = stars.as_array()
    print("Dtype after excluding:\n", stars.dtype)

    return stars


def get_mini_agasc_stars(
    filename: str,
    version_str: str,
    include_near_neighbors: bool,
) -> np.ndarray:
    """
    Reads the full AGASC data from the given file and selects the usable stars based on
    their magnitude. If `proseco` is True or `include_near_neighbors` is False, it
    includes the near-neighbor stars that got cut by the magnitude filter. Filters down
    to miniagasc stars and returns the resulting structured ndarray.

    Parameters:
    -----------
    filename : str
        The path to the file containing the full AGASC data.
    version_str : str
        The version string to use for the near-neighbor file.
    include_near_neighbors : bool
        If True, include the near-neighbor stars that got cut by the magnitude filter.

    Returns:
    --------
    np.ndarray
        The resulting array of miniagasc stars.
    """
    print(f"Reading full AGASC {filename} and selecting useable stars")

    with tables.open_file(filename) as h5:
        stars = h5.root.data[:]

    # Filter mags
    ok = stars["MAG_ACA"] - 3.0 * stars["MAG_ACA_ERR"] / 100.0 < 11.5

    # Put back near-neighbor stars that got cut by above mag filter. This file
    # is made with create_near_neighbor_ids.py.
    if proseco or include_near_neighbors:
        near_file = f"near_neighbor_ids_{version_str}.fits.gz"
        near_table = Table.read(near_file, format="fits")
        near_ids = set(near_table["near_id"])
        print(f"Including {len(near_ids)} near neighbor stars")
        for idx, agasc_id in enumerate(stars["AGASC_ID"]):
            if agasc_id in near_ids:
                ok[idx] = True

    # Filter down to miniagasc stars
    print(f"Filtering from {len(stars)} to {np.count_nonzero(ok)} stars")
    stars = stars[ok]

    return stars


def get_healpix_index_table(stars, nside) -> tuple[Table, np.ndarray]:
    """Return table that maps healpix index to row ranges in ``stars``.

    Parameters
    ----------
    stars : astropy.table.Table
        A table of stars with columns "RA" and "DEC" in degrees.
    nside : int
        The nside parameter of the HEALPix grid to use.

    Returns
    -------
    healpix_index : astropy.table.Table
        A table with columns "healpix", "row0", and "row1". The "healpix" column
        contains the healpix index for each group of rows, and the "row0" and "row1"
        columns contain the starting and ending row indices in `stars` for each group.
    idxs_sort : numpy.ndarray
        The indices that sort ``stars`` by healpix index.
    """
    hp = get_healpix(nside)

    idxs_healpix = hp.lonlat_to_healpix(stars["RA"] * u.deg, stars["DEC"] * u.deg)
    idxs_sort = np.argsort(idxs_healpix)
    idxs_healpix_sort = idxs_healpix[idxs_sort]

    # Make an index table for the positions where idxs_healpix_sort changes.
    # This is the start of each healpix.
    i0 = np.flatnonzero(np.diff(idxs_healpix_sort) != 0)
    i0s = np.concatenate([[0], i0 + 1])
    i1s = np.concatenate([i0 + 1, [len(idxs_healpix_sort)]])
    healpix_index = Table(
        [idxs_healpix_sort[i0s], i0s, i1s], names=["healpix", "row0", "row1"]
    )
    return healpix_index, idxs_sort


if __name__ == "__main__":
    main()
