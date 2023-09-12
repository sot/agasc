# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide functions for working with HEALPix-index AGASC HDF5 files.

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
    return hpx.HEALPix(nside=nside, order="nested")


@functools.lru_cache(maxsize=8)
def get_healpix_index(agasc_file):
    """
    Get the healpix index table for an AGASC file.

    The healpix index table is a table with columns ``healpix``, ``idx0`` and ``idx1``.
    This corresponds to row ranges in the main ``data`` table in the HDF5 file.

    :param agasc_file: AGASC file
    :returns: Table healpix index for catalog
    """
    with tables.open_file(agasc_file, mode="r") as h5:
        tbl = h5.root.healpix_index[:]
    out = {row["healpix"]: (row["row0"], row["row1"]) for row in tbl}

    return out


def get_stars_from_healpix_h5(
    ra: float, dec: float, radius: float, agasc_file: str | Path
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

    Returns
    -------
    stars : astropy.table.Table
        Table of stars within the search cone, with columns from the AGASC data table.
    """
    from agasc import sphere_dist

    # Table of healpix, idx0, idx1 where idx is the index into main AGASC data table
    healpix_index_map = get_healpix_index(agasc_file)

    stars_list = []
    with tables.open_file(agasc_file) as h5:
        nside = h5.root.healpix_index.attrs["nside"]
        hp = get_healpix(nside)
        # Get healpix index for ever pixel that intersects the cone.
        healpix_indices = hp.cone_search_lonlat(
            ra * u.deg, dec * u.deg, radius=radius * u.deg
        )

        for healpix_index in healpix_indices:
            idx0, idx1 = healpix_index_map[healpix_index]
            stars_list.append(h5.root.data[idx0:idx1])

    stars = Table(np.concatenate(stars_list))
    dists = sphere_dist(ra, dec, stars["RA"], stars["DEC"])
    stars = stars[dists <= radius]

    # Sort in DEC order for back-compatibility with the AGASC ordering before v1.8.
    stars.sort("DEC")

    return stars
