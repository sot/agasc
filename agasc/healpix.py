# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide functions for working with HEALPix-indexed AGASC HDF5 files.

Functions
---------
is_healpix(agasc_file)
    Return True if `agasc_file` is a HEALPix file (otherwise dec-sorted).
get_stars_from_healpix_h5(ra, dec, radius, agasc_file)
    Return a table of stars within a given radius around a given sky position.
"""

import functools
from pathlib import Path
from typing import Optional

import astropy.units as u
import astropy_healpix as hpx
import numpy as np
import tables
from astropy.table import Table


__all__ = ["is_healpix", "get_stars_from_healpix_h5"]


@functools.lru_cache()
def is_healpix(agasc_file):
    """Return True if ``agasc_file`` is a healpix file (otherwise dec-sorted)"""
    with tables.open_file(agasc_file, mode="r") as h5:
        return "healpix_index" in h5.root


@functools.lru_cache(maxsize=12)
def get_healpix(nside):
    """
    Returns a HEALPix object with the specified nside and nested order.

    Parameters
    -----------
    nside : int
        The nside parameter for the HEALPix object.

    Returns:
    --------
    hpx : HEALPix object
        A HEALPix object with the specified nside and order.
    """
    return hpx.HEALPix(nside=nside, order="nested")


@functools.lru_cache(maxsize=8)
def get_healpix_info(agasc_file: str | Path) -> tuple[dict[int, tuple[int, int]], int]:
    """
    Get the healpix index table for an AGASC file.

    The healpix index table is a table with columns ``healpix``, ``idx0`` and ``idx1``.
    This corresponds to row ranges in the main ``data`` table in the HDF5 file.

    Parameters
    ----------
    agasc_file : str or Path
        Path to the AGASC HDF5 file.

    Returns
    -------
    healpix_index : dict
        Dictionary of healpix index to row range.
    nside : int
        HEALPix nside parameter.
    """
    with tables.open_file(agasc_file, mode="r") as h5:
        tbl = h5.root.healpix_index[:]
        nside = h5.root.healpix_index.attrs["nside"]

    out = {row["healpix"]: (row["row0"], row["row1"]) for row in tbl}

    return out, nside


def get_stars_from_healpix_h5(
    ra: float,
    dec: float,
    radius: float,
    agasc_file: str | Path,
    columns: Optional[list[str] | tuple[str]] = None,
    cache: bool = False,
) -> Table:
    """
    Returns a table of stars within a given radius around a given sky position (RA, Dec),
    using the AGASC data stored in a HDF5 file with a HEALPix index.

    Parameters
    ----------
    ra : float
        Right ascension of the center of the search cone, in degrees.
    dec : float
        Declination of the center of the search cone, in degrees.
    radius : float
        Radius of the search cone, in degrees.
    agasc_file : str or Path
        Path to the HDF5 file containing the AGASC data with a HEALPix index.
    columns : list or tuple, optional
        The columns to read from the AGASC file. If not specified, all columns are read.
    cache : bool, optional
        Whether to cache the AGASC data in memory. Default is False.

    Returns
    -------
    stars : astropy.table.Table
        Table of stars within the search cone, with columns from the AGASC data table.
    """
    from agasc import sphere_dist, read_h5_table

    # Table of healpix, idx0, idx1 where idx is the index into main AGASC data table
    healpix_index_map, nside = get_healpix_info(agasc_file)
    hp = get_healpix(nside)

    # Get healpix index for ever pixel that intersects the cone.
    healpix_indices = hp.cone_search_lonlat(
        ra * u.deg, dec * u.deg, radius=radius * u.deg
    )

    stars_list = []

    def make_stars_list(h5_file):
        for healpix_index in healpix_indices:
            idx0, idx1 = healpix_index_map[healpix_index]
            stars = read_h5_table(h5_file, columns, row0=idx0, row1=idx1, cache=cache)
            stars_list.append(stars)

    if cache:
        make_stars_list(agasc_file)
    else:
        with tables.open_file(agasc_file) as h5:
            make_stars_list(h5)

    stars = Table(np.concatenate(stars_list))
    dists = sphere_dist(ra, dec, stars["RA"], stars["DEC"])
    stars = stars[dists <= radius]

    # Sort in DEC order for back-compatibility with the AGASC ordering before v1.8.
    stars.sort("DEC")

    return stars
