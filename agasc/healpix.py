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
    cache : bool, optional
        Whether to cache the AGASC data in memory. Default is False.

    Returns
    -------
    stars : astropy.table.Table
        Table of stars within the search cone, with columns from the AGASC data table.
    """
    from agasc import read_h5_table, sphere_dist

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
            stars = read_h5_table(h5_file, row0=idx0, row1=idx1, cache=cache)
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


def make_index(stars, nside, outfile=None):
    from astropy.table import Table
    from astropy import units as u
    import astropy_healpix as hpx

    hp = hpx.HEALPix(nside=nside, order="nested")
    indices = hp.lonlat_to_healpix(stars["RA"] * u.degree, stars["DEC"] * u.degree)
    if not np.all(np.diff(indices) >= 0):
        # sort by healpix index
        i = np.argsort(indices)
        indices = indices[i]
        stars = stars[i]
    assert np.all(np.diff(indices) >= 0)
    indices_diff = np.diff(indices)
    i = np.argwhere(indices_diff).flatten()
    idx = np.unique(np.concatenate([[0], i + 1, [len(indices)]]))
    index = Table()
    index["healpix"] = indices[idx[:-1]]
    index["row0"] = idx[:-1]
    index["row1"] = idx[1:]

    index = index.as_array()
    stars = np.array(stars)
    if outfile is not None:
        with tables.open_file(outfile, "w") as h5:
            h5.create_table("/", "data", stars)
            h5.create_table("/", "healpix_index", index)
            h5.root.healpix_index.attrs["nside"] = nside
    return stars, index


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
