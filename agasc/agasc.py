# Licensed under a 3-clause BSD style license - see LICENSE.rst
import contextlib
import functools
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional

import numexpr
import numpy as np
import tables
from astropy.table import Column, Table
from Chandra.Time import DateTime
from packaging.version import Version

from .healpix import get_healpix_index_table, get_stars_from_healpix_h5, is_healpix
from .paths import default_agasc_dir
from .supplement.utils import get_supplement_table

__all__ = [
    "sphere_dist",
    "get_agasc_cone",
    "get_star",
    "get_stars",
    "read_h5_table",
    "get_agasc_filename",
    "MAG_CATID_SUPPLEMENT",
    "BAD_CLASS_SUPPLEMENT",
    "set_supplement_enabled",
    "SUPPLEMENT_ENABLED_ENV",
    "write_agasc",
    "TABLE_DTYPE",
    "TableOrder",
]

logger = logging.getLogger("agasc")

SUPPLEMENT_ENABLED_ENV = "AGASC_SUPPLEMENT_ENABLED"
SUPPLEMENT_ENABLED_DEFAULT = "True"
MAG_CATID_SUPPLEMENT = 100
BAD_CLASS_SUPPLEMENT = 100

# Columns that are required for calls to get_agasc_cone
COLUMNS_REQUIRED = {
    "RA",
    "DEC",
    "EPOCH",
    "PM_DEC",
    "PM_RA",
}

RA_DECS_CACHE = {}

COMMON_AGASC_FILE_DOC = """\
If ``agasc_file`` is not specified or is None then return either
    ``default_agasc_dir()/${AGASC_HDF5_FILE}`` if ``${AGASC_HDF5_FILE}`` is defined;
    or return the latest version of ``proseco_agasc`` in ``default_agasc_dir()``.

    If ``agasc_file`` ends with the suffix ``.h5`` then it is returned as-is.

    If ``agasc_file`` ends with ``*`` then the latest version of the matching AGASC file
    in ``default_agasc_dir()`` is returned. For example, ``proseco_agasc_*`` could return
    ``${SKA}/data/agasc/proseco_agasc_1p7.h5``.

    Any other ending for ``agasc_file`` raises a ``ValueError``.

    The default AGASC directory is the environment variable ``${AGASC_DIR}`` if defined,
    otherwise ``${SKA}/data/agasc``."""

COMMON_DOC = f"""By default, stars with available mag estimates or bad star entries
    in the AGASC supplement are updated in-place in the output ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``agasc.MAG_CATID_SUPPLEMENT``.
    - If ``COLOR1`` is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.
      Those particular values do not matter except that they are different from
      the "status flag" values of 0.7 (no color => very unreliable mag estimate)
      or 1.5 (red star => somewhat unreliable mag estimate) that have special
      meaning based on deep heritage in the AGASC catalog itself.
    - If ``AGASC_ID`` is in the supplement bad stars table then CLASS is set to
      ``agasc.BAD_CLASS_SUPPLEMENT``.

    To disable the magnitude / bad star updates from the AGASC supplement, see
    the ``set_supplement_enabled`` context manager / decorator.

    {COMMON_AGASC_FILE_DOC.replace("return", "use")}

    The default AGASC supplement file is ``<AGASC_DIR>/agasc_supplement.h5``.
    """


@contextlib.contextmanager
def set_supplement_enabled(value):
    """Decorator / context manager to temporarily set the default for use of
    AGASC supplement in query functions.

    This sets the default for the ``use_supplement`` argument in AGASC function
    calls if the user does not supply that argument.

    This is mostly for testing or legacy applications to override the
    default behavior to use the AGASC supplement star mags when available.

    Examples::

      import agasc

      # Disable use of the supplement for the context block
      with agasc.set_supplement_enabled(False):
          aca = proseco.get_aca_catalog(obsid=8008)

      # Disable use of the supplement for the function
      @agasc.set_supplement_enabled(False)
      def test_get_aca_catalog():
          aca = proseco.get_aca_catalog(obsid=8008)
          ...

      # Globally disable use of the supplement everywhere
      os.environ[agasc.SUPPLEMENT_ENABLED_VAR] = 'False'

    :param value: bool
        Whether to use the AGASC supplement in the context / decorator
    """
    if not isinstance(value, bool):
        raise TypeError("value must be bool (True|False)")
    orig = os.environ.get(SUPPLEMENT_ENABLED_ENV)
    os.environ[SUPPLEMENT_ENABLED_ENV] = str(value)

    yield

    if orig is None:
        del os.environ[SUPPLEMENT_ENABLED_ENV]
    else:
        os.environ[SUPPLEMENT_ENABLED_ENV] = orig


class IdNotFound(LookupError):
    pass


class InconsistentCatalogError(Exception):
    pass


class RaDec(object):
    def __init__(self, agasc_file):
        self._agasc_file = agasc_file

    @property
    def agasc_file(self):
        return self._agasc_file

    @property
    def ra(self):
        if not hasattr(self, "_ra"):
            self._ra, self._dec = self.read_ra_dec()
        return self._ra

    @property
    def dec(self):
        if not hasattr(self, "_dec"):
            self._ra, self._dec = self.read_ra_dec()
        return self._dec

    def read_ra_dec(self):
        # Read the RA and DEC values from the agasc
        with tables.open_file(self.agasc_file) as h5:
            ras = h5.root.data.read(field="RA")
            decs = h5.root.data.read(field="DEC")
        return ras, decs


def get_ra_decs(agasc_file):
    agasc_file = os.path.abspath(agasc_file)
    if agasc_file not in RA_DECS_CACHE:
        RA_DECS_CACHE[agasc_file] = RaDec(agasc_file)
    return RA_DECS_CACHE[agasc_file]


def read_h5_table(
    h5_file: str | Path | tables.file.File,
    columns: Optional[list | tuple] = None,
    row0: Optional[int] = None,
    row1: Optional[int] = None,
    path="data",
    cache=False,
) -> np.ndarray:
    """
    Read HDF5 table ``columns`` from group ``path`` in ``h5_file``.

    If ``row0`` and ``row1`` are specified then only the rows in that range are read,
    e.g. ``data[row0:row1]``.

    If ``cache`` is ``True`` then the data for the last read is cached in memory. The
    cache key is ``(h5_file, columns, row0, row1, path)`` and only one cache entry is
    kept. It is typically not useful to cache the read if ``row0`` or ``row1`` are
    specified.

    Parameters
    ----------
    h5_file : str, Path, tables.file.File
        Path to the HDF5 file to read.
    columns : list or tuple, optional
        Column names to read from the file. If not specified, all columns are read.
    row0 : int, optional
        First row to read. Default is None (read from first row).
    row1 : int, optional
        Last row to read. Default is None (read to last row).
    path : str, optional
        Path to the data table in the HDF5 file. Default is 'data'.
    cache : bool, optional
        Whether to cache the read data. Default is False.

    Returns
    -------
    out : np.ndarray
        The HDF5 data as a numpy structured array
    """
    if columns is not None:
        columns = tuple(columns)

    if cache:
        if isinstance(h5_file, tables.file.File):
            h5_file = h5_file.filename
        data = _read_h5_table_cached(h5_file, columns, path)
        out = data[row0:row1]
    else:
        out = _read_h5_table(h5_file, columns, path, row0, row1)

    return out


@functools.lru_cache
def _read_h5_table_cached(
    h5_file: str | Path,
    columns: tuple,
    path: str,
) -> np.ndarray:
    return _read_h5_table(h5_file, columns, path, row0=None, row1=None)


def _read_h5_table(
    h5_file: str | Path | tables.file.File,
    columns: tuple,
    path: str,
    row0: None | int,
    row1: None | int,
) -> np.ndarray:
    if isinstance(h5_file, tables.file.File):
        out = _read_h5_table_from_open_h5_file(h5_file, path, row0, row1, columns)
    else:
        with tables.open_file(h5_file) as h5:
            out = _read_h5_table_from_open_h5_file(h5, path, row0, row1, columns)

    out = np.asarray(out)  # Convert to structured ndarray (not recarray)
    return out


def _read_h5_table_from_open_h5_file(
    h5: tables.file.File,
    path: str,
    row0: int,
    row1: int,
    columns: tuple | None = None,
):
    data = getattr(h5.root, path)
    out = data.read(start=row0, stop=row1)
    if columns:
        out = np.rec.fromarrays([out[col] for col in columns], names=columns)
    return np.asarray(out)


def get_agasc_filename(
    agasc_file: Optional[str | Path] = None,
    allow_rc: bool = False,
    version: Optional[str] = None,
) -> str:
    """Get a matching AGASC file name from ``agasc_file``.

    {common_agasc_file_doc}

    Parameters
    ----------
    agasc_file : str, Path, optional
        AGASC file name (default=None)
    allow_rc : bool, optional
        Allow AGASC release candidate files (default=False)
    version : str, optional
        Version number to match (e.g. "1p8" or "1p8rc4", default=None)

    Returns
    -------
    filename : str
        Matching AGASC file name

    Examples
    --------
    Setup:

    >>> from agasc import get_agasc_filename

    Selecting files in the default AGASC directory:

    >>> get_agasc_filename()
    '/Users/aldcroft/ska/data/agasc/proseco_agasc_1p7.h5'
    >>> get_agasc_filename("proseco_agasc_*")
    '/Users/aldcroft/ska/data/agasc/proseco_agasc_1p7.h5'
    >>> get_agasc_filename("proseco_agasc_*", version="1p8", allow_rc=True)
    '/Users/aldcroft/ska/data/agasc/proseco_agasc_1p8rc4.h5'
    >>> get_agasc_filename("agas*")
    Traceback (most recent call last):
       ...
    FileNotFoundError: No AGASC files in /Users/aldcroft/ska/data/agasc found matching
      agas*_?1p([0-9]+).h5

    Selecting non-default AGASC file in the default directory:

    >>> os.environ["AGASC_HDF5_FILE"] = "proseco_agasc_1p6.h5"
    >>> get_agasc_filename()
    '/Users/aldcroft/ska/data/agasc/proseco_agasc_1p6.h5'

    Changing the default AGASC directory:

    >>> os.environ["AGASC_DIR"] = "."
    >>> get_agasc_filename()
    'proseco_agasc_1p7.h5'

    Selecting an arbitrary AGASC file name either directly or with the AGASC_HDF5_FILE
    environment variable:

    >>> get_agasc_filename("any_agasc.h5")
    'any_agasc.h5'
    >>> os.environ["AGASC_HDF5_FILE"] = "whatever.h5"
    >>> get_agasc_filename()
    '/Users/aldcroft/ska/data/agasc/whatever.h5'
    """
    if agasc_file is None:
        if "AGASC_HDF5_FILE" in os.environ:
            return str(default_agasc_dir() / os.environ["AGASC_HDF5_FILE"])
        else:
            agasc_file = "proseco_agasc_*"

    agasc_file = str(agasc_file)

    if agasc_file.endswith(".h5"):
        return agasc_file

    # Get latest version of file matching agasc_file in the default AGASC dir
    agasc_dir = default_agasc_dir()

    if not agasc_file.endswith("*"):
        raise ValueError("agasc_file must end with '*' or '.h5'")

    agasc_file_re = agasc_file[:-1] + r"(1p[0-9]+) (rc[1-9][0-9]*)? \.h5$"
    matches = []
    for path in agasc_dir.glob("*.h5"):
        name = path.name
        if match := re.match(agasc_file_re, name, re.VERBOSE):
            if not allow_rc and match.group(2):
                continue
            version_str = match.group(1)
            rc_str = match.group(2) or ""
            if version is not None and version not in (
                version_str,
                version_str + rc_str,
            ):
                continue
            matches.append((Version(version_str.replace("p", ".") + rc_str), path))

    if len(matches) == 0:
        with_version = f" with {version=}" if version is not None else ""
        raise FileNotFoundError(
            f"No AGASC files in {agasc_dir}{with_version} matching {agasc_file_re}"
        )
    # Get candidate with highest version number. Tuples are sorted lexically starting
    # by first element, which is the version number here.
    out = sorted(matches)[-1][1]

    return str(out)


def sphere_dist(ra1, dec1, ra2, dec2):
    """
    Haversine formula for angular distance on a sphere: more stable at poles.
    This version uses arctan instead of arcsin and thus does better with sign
    conventions.  This uses numexpr to speed expression evaluation by a factor
    of 2 to 3.

    :param ra1: first RA (deg)
    :param dec1: first Dec (deg)
    :param ra2: second RA (deg)
    :param dec2: second Dec (deg)

    :returns: angular separation distance (deg)
    """

    ra1 = np.radians(ra1).astype(np.float64)
    ra2 = np.radians(ra2).astype(np.float64)
    dec1 = np.radians(dec1).astype(np.float64)
    dec2 = np.radians(dec2).astype(np.float64)

    numerator = numexpr.evaluate(  # noqa: F841
        "sin((dec2 - dec1) / 2) ** 2 + "
        "cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2"
    )

    dists = numexpr.evaluate("2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)")
    return np.degrees(dists)


def update_color1_column(stars: Table):
    """
    For any stars which have a V-I color (RSV3 > 0) and COLOR1 == 1.5
    then set COLOR1 = COLOR2 * 0.850.  For such stars the MAG_ACA / MAG_ACA_ERR
    values are reliable and they should not be flagged with a COLOR1 = 1.5,
    which generally implies to downstream tools that the mag is unreliable.

    Also ensure that COLOR2 values that happen to be exactly 1.5 are shifted a bit.

    The 0.850 factor is because COLOR1 = B-V while COLOR2 = BT-VT.  See
    https://heasarc.nasa.gov/W3Browse/all/tycho2.html for a reminder of the
    scaling between the two.

    This updates ``stars`` in place if the COLOR1 column is present.
    """
    if "COLOR1" not in stars.columns:
        return

    # Select red stars that have a reliable mag in AGASC 1.7 and later.
    color15 = np.isclose(stars["COLOR1"], 1.5) & (stars["RSV3"] > 0)
    new_color1 = stars["COLOR2"][color15] * 0.850

    if len(new_color1) > 0:
        # Ensure no new COLOR1 are within 0.001 of 1.5, so downstream tests of
        # COLOR1 == 1.5 or np.isclose(COLOR1, 1.5) will not accidentally succeed.
        fix15 = np.isclose(new_color1, 1.5, rtol=0, atol=0.0005)
        new_color1[fix15] = 1.499  # Insignificantly different from 1.50

        # For stars with a reliable mag, now COLOR1 is really the B-V color.
        stars["COLOR1"][color15] = new_color1


def add_pmcorr_columns(stars, date):
    """Add proper-motion corrected columns RA_PMCORR and DEC_PMCORR to Table ``stars``.

    This computes the PM-corrected RA and Dec of all ``stars`` using the supplied
    ``date``, which may be a scalar or an array-valued object (np.array or list).

    The ``stars`` table is updated in-place.

    :param stars: astropy Table of stars from the AGASC
    :param date: scalar, list, array of date(s) in DateTime-compatible format
    :returns: None
    """
    # Convert date to DateTime ensuring it can broadcast to stars table. Since
    # DateTime is slow keep it as a scalar if possible.
    if np.asarray(date).shape == ():
        dates = DateTime(date)
    else:
        dates = DateTime(np.broadcast_to(date, len(stars)))

    # Compute delta year.  stars['EPOCH'] is Column, float32. Need to coerce to
    # ndarray float64 for consistent results between scalar and array cases.
    dyear = dates.frac_year - stars["EPOCH"].view(np.ndarray).astype(np.float64)

    pm_to_degrees = dyear / (3600.0 * 1000.0)
    dec_pmcorr = np.where(
        stars["PM_DEC"] != -9999,
        stars["DEC"] + stars["PM_DEC"] * pm_to_degrees,
        stars["DEC"],
    )
    ra_scale = np.cos(np.radians(stars["DEC"]))
    ra_pmcorr = np.where(
        stars["PM_RA"] != -9999,
        stars["RA"] + stars["PM_RA"] * pm_to_degrees / ra_scale,
        stars["RA"],
    )

    # Add the proper-motion corrected columns to table using astropy.table.Table
    stars.add_columns(
        [
            Column(data=ra_pmcorr, name="RA_PMCORR"),
            Column(data=dec_pmcorr, name="DEC_PMCORR"),
        ]
    )


def get_agasc_cone(
    ra,
    dec,
    radius=1.5,
    date=None,
    agasc_file=None,
    pm_filter=True,
    fix_color1=True,
    use_supplement=None,
    columns=None,
    cache=False,
):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    The star positions are corrected for proper motion using the ``date``
    argument, which defaults to the current date if not supplied.  The
    corrected positions are available in the ``RA_PMCORR`` and ``DEC_PMCORR``
    columns, respectively.

    {common_doc}

    Example::

      >>> import agasc
      >>> stars = agasc.get_agasc_cone(10.0, 20.0, 1.5)
      >>> plt.plot(stars['RA'], stars['DEC'], '.')

    :param ra: RA (deg)
    :param dec: Declination (deg)
    :param radius: Cone search radius (deg)
    :param date: Date for proper motion (default=Now)
    :param agasc_file: AGASC file (optional)
    :param pm_filter: Use PM-corrected positions in filtering
    :param fix_color1: set COLOR1=COLOR2 * 0.85 for stars with V-I color
    :param use_supplement: Use estimated mag from AGASC supplement where available
        (default=value of AGASC_SUPPLEMENT_ENABLED env var, or True if not defined)
    :param columns: Columns to return (default=all)
    :param cache: Cache the AGASC data in memory (default=False)

    :returns: astropy Table of AGASC entries
    """
    agasc_file = get_agasc_filename(agasc_file)

    get_stars_func = (
        get_stars_from_healpix_h5
        if is_healpix(agasc_file)
        else get_stars_from_dec_sorted_h5
    )
    # Possibly expand initial radius to allow for slop due proper motion
    rad_pm = radius + (0.1 if pm_filter else 0.0)

    # Ensure that the columns we need are read from the AGASC file, excluding PMCORR
    # columns if supplied since they are not in the HDF5.
    columns_query = (
        None
        if columns is None
        else tuple(
            column for column in columns if column not in ("RA_PMCORR", "DEC_PMCORR")
        )
    )

    # Minimal columns to compute PM-corrected positions and do filtering.
    if columns and COLUMNS_REQUIRED - set(columns):
        raise ValueError(f"columns must include all of {COLUMNS_REQUIRED}")

    stars = get_stars_func(
        ra, dec, rad_pm, agasc_file=agasc_file, columns=columns_query, cache=cache
    )

    add_pmcorr_columns(stars, date)
    if fix_color1:
        update_color1_column(stars)

    # Final filtering using proper-motion corrected positions
    if pm_filter:
        dists = sphere_dist(ra, dec, stars["RA_PMCORR"], stars["DEC_PMCORR"])
        ok = dists <= radius
        stars = stars[ok]

    update_from_supplement(stars, use_supplement)

    stars.meta["agasc_file"] = agasc_file
    return stars


def get_stars_from_dec_sorted_h5(
    ra: float,
    dec: float,
    radius: float,
    agasc_file: str | Path,
    columns: Optional[list[str] | tuple[str]] = None,
    cache: bool = False,
) -> Table:
    """
    Returns a table of stars within a given radius of a given RA and Dec.

    Parameters
    ----------
    ra : float
        The right ascension of the center of the search radius, in degrees.
    dec : float
        The declination of the center of the search radius, in degrees.
    radius : float
        The radius of the search circle, in degrees.
    agasc_file : str or Path
        The path to the AGASC HDF5 file.
    columns : list or tuple, optional
        The columns to read from the AGASC file. If not specified, all columns are read.
    cache : bool, optional
        Whether to cache the AGASC data in memory. Default is False.

    Returns
    -------
    stars : astropy.table.Table
        A structured ndarray of stars within the search radius, sorted by declination.
    """
    ra_decs = get_ra_decs(agasc_file)
    idx0, idx1 = np.searchsorted(ra_decs.dec, [dec - radius, dec + radius])

    dists = sphere_dist(ra, dec, ra_decs.ra[idx0:idx1], ra_decs.dec[idx0:idx1])
    ok = dists <= radius

    stars = read_h5_table(agasc_file, columns, row0=idx0, row1=idx1, cache=cache)
    stars = Table(stars[ok])

    return stars


def get_star(id, agasc_file=None, date=None, fix_color1=True, use_supplement=None):
    """
    Get AGASC catalog entry for star with requested id.

    {common_doc}

    Example::

      >>> import agasc
      >>> star = agasc.get_star(636629880)
      >>> for name in star.colnames:
      ...     print '{{:12s}} : {{}}'.format(name, star[name])
      AGASC_ID     : 636629880
      RA           : 125.64184
      DEC          : -4.23235
      POS_ERR      : 300
      POS_CATID    : 6
      EPOCH        : 1983.0
      PM_RA        : -9999
      PM_DEC       : -9999
      PM_CATID     : 0
      PLX          : -9999
      PLX_ERR      : -9999
      PLX_CATID    : 0
      MAG_ACA      : 12.1160011292
      MAG_ACA_ERR  : 45
      CLASS        : 0
      MAG          : 13.2700004578
      ...

    :param id: AGASC id
    :param date: Date for proper motion (default=Now)
    :param fix_color1: set COLOR1=COLOR2 * 0.85 for stars with V-I color (default=True)
    :param use_supplement: Use estimated mag from AGASC supplement where available
        (default=value of AGASC_SUPPLEMENT_ENABLED env var, or True if not defined)
    :returns: astropy Table Row of entry for id
    """

    agasc_file = get_agasc_filename(agasc_file)

    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data
        id_rows = tbl.read_where("(AGASC_ID == {})".format(id))

    if len(id_rows) > 1:
        raise InconsistentCatalogError(
            "More than one entry found for {} in AGASC".format(id)
        )

    if id_rows is None or len(id_rows) == 0:
        raise IdNotFound()

    t = Table(id_rows)
    add_pmcorr_columns(t, date)
    if fix_color1:
        update_color1_column(t)

    update_from_supplement(t, use_supplement)

    return t[0]


def _get_rows_read_where(ids_1d, dates_1d, agasc_file):
    rows = []
    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data
        for id, _date in zip(ids_1d, dates_1d):
            id_rows = tbl.read_where("(AGASC_ID == {})".format(id))

            if len(id_rows) > 1:
                raise InconsistentCatalogError(
                    f"More than one entry found for {id} in AGASC"
                )

            if id_rows is None or len(id_rows) == 0:
                raise IdNotFound(f"No entry found for {id} in AGASC")

            rows.append(id_rows[0])
    return rows


def _get_rows_read_entire(ids_1d, dates_1d, agasc_file):
    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data[:]

    agasc_idx = {agasc_id: idx for idx, agasc_id in enumerate(tbl["AGASC_ID"])}

    rows = []
    for agasc_id, _date in zip(ids_1d, dates_1d):
        if agasc_id not in agasc_idx:
            raise IdNotFound(f"No entry found for {agasc_id} in AGASC")

        rows.append(tbl[agasc_idx[agasc_id]])
    return rows


def get_stars(
    ids,
    agasc_file=None,
    dates=None,
    method_threshold=5000,
    fix_color1=True,
    use_supplement=None,
):
    """
    Get AGASC catalog entries for star ``ids`` at ``dates``.

    The input ``ids`` and ``dates`` are broadcast together for the output shape
    (though note that the result is flattened in the end). If both are scalar
    inputs then the output is a Table Row, otherwise the output is a Table.

    This function has two possible methods for getting stars, either by using
    the HDF5 ``tables.read_where()`` function to get one star at a time from the
    HDF5 file, or by reading the entire table into memory and doing the search
    by making a dict index by AGASC ID. Tests indicate that the latter is faster
    for about 5000 or more stars, so this function will read the entire AGASC if
    the number of stars is greater than ``method_threshold``.

    Unlike the similar ``get_star`` function, this adds a ``DATE`` column
    indicating the date at which the star coordinates (RA_PMCORR, DEC_PMCORR)
    are computed.

    {common_doc}

    Example::
      >>> import agasc
      >>> star = agasc.get_stars(636629880)
      >>> for name in star.colnames:
      ...     print '{{:12s}} : {{}}'.format(name, star[name])
      AGASC_ID     : 636629880
      RA           : 125.64184
      DEC          : -4.23235
      POS_ERR      : 300
      POS_CATID    : 6
      EPOCH        : 1983.0
      PM_RA        : -9999
      PM_DEC       : -9999
      PM_CATID     : 0
      PLX          : -9999
      PLX_ERR      : -9999
      PLX_CATID    : 0
      MAG_ACA      : 12.1160011292
      MAG_ACA_ERR  : 45
      CLASS        : 0
      MAG          : 13.2700004578
      ...

    :param ids: AGASC ids (scalar or array)
    :param dates: Dates for proper motion (scalar or array) (default=Now)
    :param fix_color1: set COLOR1=COLOR2 * 0.85 for stars with V-I color (default=True)
    :param use_supplement: Use estimated mag from AGASC supplement where available
        (default=value of AGASC_SUPPLEMENT_ENABLED env var, or True if not defined)
    :returns: astropy Table of AGASC entries, or Table Row of one entry for scalar input
    """
    agasc_file = get_agasc_filename(agasc_file)

    dates_in = DateTime(dates).date
    dates_is_scalar = np.asarray(dates_in).shape == ()

    ids, dates = np.broadcast_arrays(ids, dates_in)
    ids_1d, dates_1d = np.atleast_1d(ids), np.atleast_1d(dates)

    if len(ids_1d) < method_threshold:
        rows = _get_rows_read_where(ids_1d, dates_1d, agasc_file)
        method = "tables_read_where"
    else:
        rows = _get_rows_read_entire(ids_1d, dates_1d, agasc_file)
        method = "read_entire_agasc"

    t = Table(np.vstack(rows).flatten())

    # Define a temporary attribute indicating get_stars method, mostly for testing
    t.get_stars_method = method

    add_pmcorr_columns(t, dates_in if dates_is_scalar else dates)
    if fix_color1:
        update_color1_column(t)
    t["DATE"] = dates

    update_from_supplement(t, use_supplement)

    return t if ids.shape else t[0]


# Interpolate common docs into function docstrings. Using f-string interpolation in the
# docstring itself does not work.
for func in get_stars, get_star, get_agasc_cone:
    func.__doc__ = func.__doc__.format(common_doc=COMMON_DOC)
get_agasc_filename.__doc__ = get_agasc_filename.__doc__.format(
    common_agasc_file_doc=COMMON_AGASC_FILE_DOC
)


def update_from_supplement(stars, use_supplement=None):
    """Update mag, color1 and class information from AGASC supplement in ``stars``.

    Stars with available mag estimates in the AGASC supplement are updated
    in-place in the input ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT``.
    - If COLOR1 is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.
      Those particular values do not matter except that they are different from
      the "status flag" values of 0.7 (no color => very unreliable mag estimate)
      or 1.5 (red star => somewhat unreliable mag estimate) that have special
      meaning based on deep heritage in the AGASC catalog itself.

    Stars which are in the bad stars table are updated as follows:

    - ``CLASS = BAD_CLASS_SUPPLEMENT``

    Whether to actually apply the update is set by a combination of the
    ``use_supplement`` argument, which has priority, and the
    ``AGASC_SUPPLEMENT_ENABLED`` environment variable.

    :param stars: astropy.table.Table of stars
    :param use_supplement: bool, None
        Use the supplement (default=None, see above)
    """
    if use_supplement is None:
        supplement_enabled_env = os.environ.get(
            SUPPLEMENT_ENABLED_ENV, SUPPLEMENT_ENABLED_DEFAULT
        )
        if supplement_enabled_env not in ("True", "False"):
            raise ValueError(
                f'{SUPPLEMENT_ENABLED_ENV} env var must be either "True" or "False" '
                f"got {supplement_enabled_env}"
            )
        supplement_enabled = supplement_enabled_env == "True"
    else:
        supplement_enabled = use_supplement

    if not supplement_enabled or len(stars) == 0:
        return

    def set_star(star, name, value):
        """Set star[name] = value if ``name`` is a column in the table"""
        try:
            star[name] = value
        except KeyError:
            pass

    # Get estimate mags and errs from supplement as a dict of dict
    # agasc_id : {mag_aca: .., mag_aca_err: ..}.
    supplement_mags = get_supplement_table("mags", agasc_dir=default_agasc_dir())
    supplement_mags_index = supplement_mags.meta["index"]

    # Get bad stars as {agasc_id: {source: ..}}
    bad_stars = get_supplement_table("bad", agasc_dir=default_agasc_dir())
    bad_stars_index = bad_stars.meta["index"]

    for star in stars:
        agasc_id = int(star["AGASC_ID"])
        if agasc_id in supplement_mags_index:
            idx = supplement_mags_index[agasc_id]
            mag_est = supplement_mags["mag_aca"][idx]
            mag_est_err = supplement_mags["mag_aca_err"][idx]

            set_star(star, "MAG_ACA", mag_est)
            # Mag err is stored as int16 in units of 0.01 mag. Use same convention here.
            set_star(star, "MAG_ACA_ERR", round(mag_est_err * 100))
            set_star(star, "MAG_CATID", MAG_CATID_SUPPLEMENT)
            if "COLOR1" in stars.colnames:
                color1 = star["COLOR1"]
                if np.isclose(color1, 0.7) or np.isclose(color1, 1.5):
                    star["COLOR1"] = color1 - 0.01

        if agasc_id in bad_stars_index:
            set_star(star, "CLASS", BAD_CLASS_SUPPLEMENT)


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


import numpy.typing as npt

TABLE_DTYPES: dict[str, npt.DTypeLike] = {
    "AGASC_ID": np.int32,
    "RA": np.float64,
    "DEC": np.float64,
    "POS_ERR": np.int16,
    "POS_CATID": np.uint8,
    "EPOCH": np.float32,
    "PM_RA": np.int16,
    "PM_DEC": np.int16,
    "PM_CATID": np.uint8,
    "PLX": np.int16,
    "PLX_ERR": np.int16,
    "PLX_CATID": np.uint8,
    "MAG_ACA": np.float32,
    "MAG_ACA_ERR": np.int16,
    "CLASS": np.int16,
    "MAG": np.float32,
    "MAG_ERR": np.int16,
    "MAG_BAND": np.int16,
    "MAG_CATID": np.uint8,
    "COLOR1": np.float32,
    "COLOR1_ERR": np.int16,
    "C1_CATID": np.uint8,
    "COLOR2": np.float32,
    "COLOR2_ERR": np.int16,
    "C2_CATID": np.uint8,
    "RSV1": np.float32,
    "RSV2": np.int16,
    "RSV3": np.uint8,
    "VAR": np.int16,
    "VAR_CATID": np.uint8,
    "ASPQ1": np.int16,
    "ASPQ2": np.int16,
    "ASPQ3": np.int16,
    "ACQQ1": np.int16,
    "ACQQ2": np.int16,
    "ACQQ3": np.int16,
    "ACQQ4": np.int16,
    "ACQQ5": np.int16,
    "ACQQ6": np.int16,
    "XREF_ID1": np.int32,
    "XREF_ID2": np.int32,
    "XREF_ID3": np.int32,
    "XREF_ID4": np.int32,
    "XREF_ID5": np.int32,
    "RSV4": np.int16,
    "RSV5": np.int16,
    "RSV6": np.int16,
}


TABLE_DTYPE = np.dtype(list(TABLE_DTYPES.items()))
"""Standard dtypes for AGASC table."""


class TableOrder(Enum):
    """
    Enumeration type to specify the AGASC table ordering:

    - TableOrder.NONE. The stars are not sorted.
    - TableOrder.HEALPIX. The stars are sorted using a HEALPix index.
    - TableOrder.DEC. The stars are sorted by declination.

    """

    NONE = 1
    DEC = 2
    HEALPIX = 3


def write_agasc(
    filename: str,
    stars: np.ndarray,
    version: str,
    nside=64,
    order=TableOrder.HEALPIX,
    full_agasc=True,
):
    """
    Write AGASC stars to a new HDF5 file.

    The table is coerced to the correct dtype if necessary (:any:`TABLE_DTYPE`).

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to write to.
    stars : np.ndarray
        The AGASC stars to write.
    version : str
        The AGASC version number. This sets an attribute in the table.
    nside : int, optional
        The HEALPix NSIDE parameter to use for the HEALPix index table.
        Default is 64.
    order : :any:`TableOrder`, optional
        The order of the stars in the AGASC file (Default is TableOrder.HEALPIX).
        The options are:

            - TableOrder.HEALPIX. The stars are sorted using a HEALPix index.
            - TableOrder.DEC. The stars are sorted by declination.
    full_agasc : bool, optional
        Whether writing a full AGASC table with all columns or a subset (normally
        proseco). Default is True, in which case all AGASC columns are required.
    """
    star_cols_set = set(stars.dtype.names)
    table_dtypes_list = list(TABLE_DTYPES.items())

    if stars.dtype != np.dtype(table_dtypes_list):
        if disallowed_keys := star_cols_set - set(TABLE_DTYPES):
            # Stars has keys that are not in allowed set
            raise ValueError(f"stars has disallowed keys: {disallowed_keys}")

        if full_agasc:
            if missing_keys := set(TABLE_DTYPES) - star_cols_set:
                # Stars is missing some keys in required set
                raise ValueError(f"missing keys in stars: {missing_keys}")
            stars_out_dtypes = table_dtypes_list
        else:
            # Allow for a subset of AGASC columns in stars (e.g. proseco_agasc) making
            # sure to preserve standard AGASC column ordering and dtype.
            stars_out_dtypes = [
                (col, dtype) for col, dtype in table_dtypes_list if col in star_cols_set
            ]

        # Coerce stars structured ndarray to the correct column ordering and dtype with
        # a side trip through astropy Table.
        cols = [col for col, _ in stars_out_dtypes]
        stars = Table(stars)[cols].as_array().astype(stars_out_dtypes)

    match order:
        case TableOrder.DEC:
            logger.info("Sorting on DEC column")
            idx_sort = np.argsort(stars["DEC"])
        case TableOrder.HEALPIX:
            logger.info(
                f"Creating healpix_index table for nside={nside} "
                "and sorting by healpix index"
            )
            healpix_index, idx_sort = get_healpix_index_table(stars, nside)
    stars = stars.take(idx_sort)

    _write_agasc(filename, stars, version)
    if order == TableOrder.HEALPIX:
        write_healpix_index_table(filename, healpix_index, nside)


def _write_agasc(filename, stars, version):
    """Do the actual writing to HDF file."""
    logger.info(f"Creating {filename}")

    with tables.open_file(filename, mode="w") as h5:
        data = h5.create_table("/", "data", stars, title=f"AGASC {version}")
        data.attrs["version"] = version
        data.flush()

        logger.info("  Creating AGASC_ID index")
        data.cols.AGASC_ID.create_csindex()

        logger.info(f"  Flush and close {filename}")
        data.flush()
        h5.flush()
