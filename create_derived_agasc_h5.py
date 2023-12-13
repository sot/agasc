# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This script creates an AGASC HDF5 file which is derived from the full AGASC catalog.

There are three key common options for creating the new AGASC file:

- ``--filter-faint``: Filter out faint stars satisfying
  ``stars["MAG_ACA"] - 3.0 * stars["MAG_ACA_ERR"] / 100.0 >= 11.5``.
- ``--include-near-neighbors``: Include near-neighbor stars even if filtered out by
    by ``--filter-faint`.
- ``--proseco-columns``: Include only columns needed for proseco.

These can be combined in various ways to create different AGASC files, but in practice
the following AGASC flavors are used with options:

- "proseco_agasc": ``--filter-faint``, ``--include-near-neighbors``, ``--proseco-columns``
- "miniagasc": ``--filter-faint``

The script depends on the file ``agasc<version>_near_neighbor_ids.fits.gz`` which is
created using ``create_near_neighbor_ids.py``.

Since version 1.8, AGASC files should be sorted by HEALPix index and include a
``healpix_index`` table.

Examples
--------

  $ python create_derived_agasc_h5.py proseco_agasc --version 1.8 \
      --filter-faint --include-near-neighbors --proseco-columns
  $ python create_derived_agasc_h5.py miniagasc --version 1.8 --filter-faint
  $ python create_derived_agasc_h5.py agasc_healpix --version 1.7
"""


import argparse
from pathlib import Path

import numpy as np
import tables
from astropy.table import Table

from agasc import default_agasc_dir, Order, write_agasc


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create derived AGASC HDF5 file", usage=__doc__
    )
    parser.add_argument(
        "out_root",
        type=str,
        help="Output filename root to create <out_root>_<version>.h5",
    )
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
        "--filter-faint",
        action="store_true",
        help='Filter: stars["MAG_ACA"] - 3.0 * stars["MAG_ACA_ERR"] / 100.0 < 11.5 ',
    )
    parser.add_argument(
        "--proseco-columns",
        action="store_true",
        help="Include only columns needed for proseco",
    )
    parser.add_argument(
        "--dec-order",
        action="store_true",
        help="Create legacy AGASC file using Dec ordering (default is HEALpix)",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="HEALPix nside parameter (default=64)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    version_num = args.version
    version = version_num.replace(".", "p")
    rootname = args.out_root

    filename_full = default_agasc_dir() / f"agasc{version}.h5"
    filename_derived = f"{rootname}_{version}.h5"

    stars = get_derived_agasc_stars(
        filename_full,
        filter_faint=args.filter_faint,
        include_near_neighbors=args.include_near_neighbors,
    )

    if args.proseco_columns:
        stars = filter_proseco_columns(stars)

    if args.dec_order:
        print("Will sort on DEC column")
        order = Order.DEC
    else:
        order = Order.HEALPIX
        print(
            f"Will create healpix_index table for nside={args.nside} "
            "and sort by healpix index"
        )

    write_agasc(filename_derived, stars, version, nside=args.nside, order=order)


def filter_proseco_columns(stars):
    print("Excluding columns not needed for proseco")
    # fmt: off
    excludes = ["PLX", "PLX_ERR", "PLX_CATID",
                "ACQQ1", "ACQQ2", "ACQQ3", "ACQQ4", "ACQQ5", "ACQQ6",
                "XREF_ID1", "XREF_ID2", "XREF_ID3", "XREF_ID4", "XREF_ID5",
                "RSV4", "RSV5", "RSV6",
                "POS_CATID", "PM_CATID",
                "MAG", "MAG_ERR", "MAG_BAND", "MAG_CATID",
                "COLOR1_ERR", "C1_CATID",  # Keep color1, 2, 3
                "COLOR2_ERR", "C2_CATID",
                "RSV2",
                "VAR_CATID"]
    # fmt: on

    names = [name for name in stars.dtype.names if name not in excludes]
    stars = Table({name: stars[name] for name in names}, copy=False)
    stars = stars.as_array()

    return stars


def get_derived_agasc_stars(
    agasc_full: str,
    filter_faint: bool,
    include_near_neighbors: bool,
) -> np.ndarray:
    """
    Reads the full AGASC data from the given file and selects the usable stars based on
    their magnitude. If `proseco` is True or `include_near_neighbors` is False, it
    includes the near-neighbor stars that got cut by the magnitude filter. Filters down
    to derived AGASC stars and returns the resulting structured ndarray.

    Parameters:
    -----------
    filename : str
        The path to the file containing the full AGASC data.
    filter_faint : bool
        If True, filter faint stars
    include_near_neighbors : bool
        If True, include the near-neighbor stars that got cut by the magnitude filter.

    Returns:
    --------
    np.ndarray
        The resulting array of derived AGASC stars.
    """
    print(f"Reading full AGASC {agasc_full} and selecting useable stars")

    with tables.open_file(agasc_full) as h5:
        stars = h5.root.data[:]

    # Filter mags
    ok = np.ones(len(stars), dtype=bool)

    if filter_faint:
        ok &= stars["MAG_ACA"] - 3.0 * stars["MAG_ACA_ERR"] / 100.0 < 11.5

    # Put back near-neighbor stars that got cut by above mag filter. This file
    # is made with create_near_neighbor_ids.py.
    if include_near_neighbors:
        agasc_full_name = Path(agasc_full).with_suffix("").name
        near_file = f"{agasc_full_name}_near_neighbor_ids.fits.gz"
        near_table = Table.read(near_file, format="fits")
        near_ids = set(near_table["near_id"])
        print(f"Including {len(near_ids)} near neighbor stars")
        for idx, agasc_id in enumerate(stars["AGASC_ID"]):
            if agasc_id in near_ids:
                ok[idx] = True

    # Filter down to derived agasc stars
    print(f"Filtering from {len(stars)} to {np.count_nonzero(ok)} stars")
    stars = stars[ok]

    return stars


if __name__ == "__main__":
    main()
