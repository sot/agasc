# Licensed under a 3-clause BSD style license - see LICENSE.rst
import contextlib
import functools
import os
from pathlib import Path
from typing import Optional

import numexpr
import numpy as np
import tables
from astropy.table import Column, Table
from Chandra.Time import DateTime

from .healpix import get_stars_from_healpix_h5, is_healpix
from .paths import default_agasc_dir, default_agasc_file
from .supplement.utils import get_supplement_table

__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star', 'get_stars', 'read_h5_table',
           'MAG_CATID_SUPPLEMENT', 'BAD_CLASS_SUPPLEMENT',
           'set_supplement_enabled', 'SUPPLEMENT_ENABLED_ENV']

SUPPLEMENT_ENABLED_ENV = 'AGASC_SUPPLEMENT_ENABLED'
SUPPLEMENT_ENABLED_DEFAULT = 'True'
MAG_CATID_SUPPLEMENT = 100
BAD_CLASS_SUPPLEMENT = 100

RA_DECS_CACHE = {}

COMMON_DOC = """By default, stars with available mag estimates or bad star entries
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

    The default ``agasc_file`` is ``<AGASC_DIR>/miniagasc.h5``, where
    ``<AGASC_DIR>`` is either the ``AGASC_DIR`` environment variable if defined
    or ``$SKA/data/agasc``.

    The default AGASC supplement file is ``<AGASC_DIR>/agasc_supplement.h5``."""


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
        raise TypeError('value must be bool (True|False)')
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
        if not hasattr(self, '_ra'):
            self._ra, self._dec = self.read_ra_dec()
        return self._ra

    @property
    def dec(self):
        if not hasattr(self, '_dec'):
            self._ra, self._dec = self.read_ra_dec()
        return self._dec

    def read_ra_dec(self):
        # Read the RA and DEC values from the agasc
        with tables.open_file(self.agasc_file) as h5:
            ras = h5.root.data.read(field='RA')
            decs = h5.root.data.read(field='DEC')
        return ras, decs


def get_ra_decs(agasc_file):
    agasc_file = os.path.abspath(agasc_file)
    if agasc_file not in RA_DECS_CACHE:
        RA_DECS_CACHE[agasc_file] = RaDec(agasc_file)
    return RA_DECS_CACHE[agasc_file]


def read_h5_table(
        h5_file: str | Path | tables.file.File,
        row0: Optional[int] = None,
        row1: Optional[int] = None,
        path="data",
        cache=False,
    ) -> np.ndarray:
    """
    Read HDF5 table from group ``path`` in ``h5_file``.

    If ``row0`` and ``row1`` are specified then only the rows in that range are read,
    e.g. ``data[row0:row1]``.

    If ``cache`` is ``True`` then the data for the last read is cached in memory. The
    cache key is ``(h5_file, path)`` and only one cache entry is kept. If ``h5_file``
    is an HDF5 file object then the filename is used as the cache key.

    Parameters
    ----------
    h5_file : str, Path, tables.file.File
        Path to the HDF5 file to read or an open HDF5 file from ``tables.open_file``.
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
    if cache:
        if isinstance(h5_file, tables.file.File):
            h5_file = h5_file.filename
        data = _read_h5_table_cached(h5_file, path)
        out = data[row0:row1]
    else:
        out = _read_h5_table(h5_file, path, row0, row1)

    return out


@functools.lru_cache(maxsize=1)
def _read_h5_table_cached(
    h5_file: str | Path,
    path: str,
) -> np.ndarray:
    return _read_h5_table(h5_file, path, row0=None, row1=None)


def _read_h5_table(
        h5_file: str | Path | tables.file.File,
        path: str,
        row0: None | int,
        row1: None | int,
    ) -> np.ndarray:
    if isinstance(h5_file, tables.file.File):
        out = _read_h5_table_from_open_h5_file(h5_file, path, row0, row1)
    else:
        with tables.open_file(h5_file) as h5:
            out = _read_h5_table_from_open_h5_file(h5, path, row0, row1)

    out = np.asarray(out)  # Convert to structured ndarray (not recarray)
    return out

def _read_h5_table_from_open_h5_file(h5, path, row0, row1):
    data = getattr(h5.root, path)
    out = data.read(start=row0, stop=row1)
    return out


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

    numerator = numexpr.evaluate('sin((dec2 - dec1) / 2) ** 2 + '  # noqa
                                 'cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2')

    dists = numexpr.evaluate('2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)')
    return np.degrees(dists)


def update_color1_column(stars):
    """
    For any stars which have a V-I color (RSV3 > 0) and COLOR1 == 1.5
    then set COLOR1 = COLOR2 * 0.850.  For such stars the MAG_ACA / MAG_ACA_ERR
    values are reliable and they should not be flagged with a COLOR1 = 1.5,
    which generally implies to downstream tools that the mag is unreliable.

    Also ensure that COLOR2 values that happen to be exactly 1.5 are shifted a bit.

    The 0.850 factor is because COLOR1 = B-V while COLOR2 = BT-VT.  See
    https://heasarc.nasa.gov/W3Browse/all/tycho2.html for a reminder of the
    scaling between the two.

    This updates ``stars`` in place.
    """
    # Select red stars that have a reliable mag in AGASC 1.7 and later.
    color15 = np.isclose(stars['COLOR1'], 1.5) & (stars['RSV3'] > 0)
    new_color1 = stars['COLOR2'][color15] * 0.850

    if len(new_color1) > 0:
        # Ensure no new COLOR1 are within 0.001 of 1.5, so downstream tests of
        # COLOR1 == 1.5 or np.isclose(COLOR1, 1.5) will not accidentally succeed.
        fix15 = np.isclose(new_color1, 1.5, rtol=0, atol=0.0005)
        new_color1[fix15] = 1.499  # Insignificantly different from 1.50

        # For stars with a reliable mag, now COLOR1 is really the B-V color.
        stars['COLOR1'][color15] = new_color1


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
    dyear = dates.frac_year - stars['EPOCH'].view(np.ndarray).astype(np.float64)

    pm_to_degrees = dyear / (3600. * 1000.)
    dec_pmcorr = np.where(stars['PM_DEC'] != -9999,
                          stars['DEC'] + stars['PM_DEC'] * pm_to_degrees,
                          stars['DEC'])
    ra_scale = np.cos(np.radians(stars['DEC']))
    ra_pmcorr = np.where(stars['PM_RA'] != -9999,
                         stars['RA'] + stars['PM_RA'] * pm_to_degrees / ra_scale,
                         stars['RA'])

    # Add the proper-motion corrected columns to table using astropy.table.Table
    stars.add_columns([Column(data=ra_pmcorr, name='RA_PMCORR'),
                       Column(data=dec_pmcorr, name='DEC_PMCORR')])


def get_agasc_cone(ra, dec, radius=1.5, date=None, agasc_file=None,
                   pm_filter=True, fix_color1=True, use_supplement=None,
                   cache=False):
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
    :param agasc_file: Mini-agasc HDF5 file sorted by Dec (optional)
    :param pm_filter: Use PM-corrected positions in filtering
    :param fix_color1: set COLOR1=COLOR2 * 0.85 for stars with V-I color
    :param use_supplement: Use estimated mag from AGASC supplement where available
        (default=value of AGASC_SUPPLEMENT_ENABLED env var, or True if not defined)
    :param cache: Cache the AGASC data in memory (default=False)

    :returns: astropy Table of AGASC entries
    """
    if agasc_file is None:
        agasc_file = default_agasc_file()

    get_stars_func = (
        get_stars_from_healpix_h5
        if is_healpix(agasc_file)
        else get_stars_from_dec_sorted_h5
    )
    # Possibly expand initial radius to allow for slop due proper motion
    rad_pm = radius + (0.1 if pm_filter else 0.0)

    stars = get_stars_func(ra, dec, rad_pm, agasc_file, cache)
    add_pmcorr_columns(stars, date)
    if fix_color1:
        update_color1_column(stars)

    # Final filtering using proper-motion corrected positions
    if pm_filter:
        dists = sphere_dist(ra, dec, stars['RA_PMCORR'], stars['DEC_PMCORR'])
        ok = dists <= radius
        stars = stars[ok]

    update_from_supplement(stars, use_supplement)

    return stars


def get_stars_from_dec_sorted_h5(
        ra: float,
        dec: float,
        radius: float,
        agasc_file: str | Path,
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

    stars = read_h5_table(agasc_file, row0=idx0, row1=idx1, cache=cache)
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

    if agasc_file is None:
        agasc_file = default_agasc_file()

    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data
        id_rows = tbl.read_where('(AGASC_ID == {})'.format(id))

    if len(id_rows) > 1:
        raise InconsistentCatalogError(
            "More than one entry found for {} in AGASC".format(id))

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
        for id, date in zip(ids_1d, dates_1d):
            id_rows = tbl.read_where('(AGASC_ID == {})'.format(id))

            if len(id_rows) > 1:
                raise InconsistentCatalogError(
                    f'More than one entry found for {id} in AGASC')

            if id_rows is None or len(id_rows) == 0:
                raise IdNotFound(f'No entry found for {id} in AGASC')

            rows.append(id_rows[0])
    return rows


def _get_rows_read_entire(ids_1d, dates_1d, agasc_file):
    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data[:]

    agasc_idx = {agasc_id: idx for idx, agasc_id in enumerate(tbl['AGASC_ID'])}

    rows = []
    for agasc_id, date in zip(ids_1d, dates_1d):
        if agasc_id not in agasc_idx:
            raise IdNotFound(f'No entry found for {agasc_id} in AGASC')

        rows.append(tbl[agasc_idx[agasc_id]])
    return rows


def get_stars(ids, agasc_file=None, dates=None, method_threshold=5000, fix_color1=True,
              use_supplement=None):
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

    if agasc_file is None:
        agasc_file = default_agasc_file()

    dates_in = DateTime(dates).date
    dates_is_scalar = np.asarray(dates_in).shape == ()

    ids, dates = np.broadcast_arrays(ids, dates_in)
    ids_1d, dates_1d = np.atleast_1d(ids), np.atleast_1d(dates)

    if len(ids_1d) < method_threshold:
        rows = _get_rows_read_where(ids_1d, dates_1d, agasc_file)
        method = 'tables_read_where'
    else:
        rows = _get_rows_read_entire(ids_1d, dates_1d, agasc_file)
        method = 'read_entire_agasc'

    t = Table(np.vstack(rows).flatten())

    # Define a temporary attribute indicating get_stars method, mostly for testing
    t.get_stars_method = method

    add_pmcorr_columns(t, dates_in if dates_is_scalar else dates)
    if fix_color1:
        update_color1_column(t)
    t['DATE'] = dates

    update_from_supplement(t, use_supplement)

    return t if ids.shape else t[0]


# Interpolate COMMON_DOC into those function docstrings
for func in get_stars, get_star, get_agasc_cone:
    func.__doc__ = func.__doc__.format(common_doc=COMMON_DOC)


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
        supplement_enabled_env = os.environ.get(SUPPLEMENT_ENABLED_ENV, SUPPLEMENT_ENABLED_DEFAULT)
        if supplement_enabled_env not in ('True', 'False'):
            raise ValueError(f'{SUPPLEMENT_ENABLED_ENV} env var must be either "True" or "False" '
                             f'got {supplement_enabled_env}')
        supplement_enabled = supplement_enabled_env == 'True'
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
    supplement_mags = get_supplement_table('mags', agasc_dir=default_agasc_dir(),
                                           as_dict=True)

    # Get bad stars as {agasc_id: {source: ..}}
    bad_stars = get_supplement_table('bad', agasc_dir=default_agasc_dir(), as_dict=True)

    for star in stars:
        agasc_id = int(star['AGASC_ID'])
        if agasc_id in supplement_mags:
            mag_est = supplement_mags[agasc_id]['mag_aca']
            mag_est_err = supplement_mags[agasc_id]['mag_aca_err']

            set_star(star, 'MAG_ACA', mag_est)
            # Mag err is stored as int16 in units of 0.01 mag. Use same convention here.
            set_star(star, 'MAG_ACA_ERR', round(mag_est_err * 100))
            set_star(star, 'MAG_CATID', MAG_CATID_SUPPLEMENT)
            if 'COLOR1' in stars.colnames:
                color1 = star['COLOR1']
                if np.isclose(color1, 0.7) or np.isclose(color1, 1.5):
                    star['COLOR1'] = color1 - 0.01

        if agasc_id in bad_stars:
            set_star(star, 'CLASS', BAD_CLASS_SUPPLEMENT)
