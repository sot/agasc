# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path
import warnings
import functools

import numpy as np
import numexpr
import tables

from Chandra.Time import DateTime
from astropy.table import Table, Column
from ska_helpers.utils import lru_cache_timed

__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star', 'get_stars',
           'MAG_CATID_SUPPLEMENT', 'get_supplement_table',
           'disable_supplement']

DISABLE_SUPPLEMENT_ENV = 'AGASC_DISABLE_SUPPLEMENT'
MAG_CATID_SUPPLEMENT = 100
RA_DECS_CACHE = {}
COMMON_DOC = """\
If ``use_supplement`` is ``True`` and there is no ``AGASC_DISABLE_SUPPLEMENT``
    environment variable set, then stars with available mag estimates or bad
    star entries in the AGASC supplement are updated in-place in the output
    ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT`` (100).
    - If ``COLOR1`` is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.
    - If ``AGASC_ID`` is in the supplement bad stars table then CLASS is set to
      ``100 + <source>``, where ``<source>`` is the bad star source identifier.

    The default ``agasc_file`` is ``<AGASC_DIR>/miniagasc.h5``, where
    ``<AGASC_DIR>`` is either the ``AGASC_DIR`` environment variable if defined
    or ``$SKA/data/agasc``."""


def default_agasc_dir():
    """Path to the AGASC directory.

    This returns the ``AGASC_DIR`` environment variable if defined, otherwise
    ``$SKA/data/agasc``.

    :returns: Path
    """
    if 'AGASC_DIR' in os.environ:
        out = Path(os.environ['AGASC_DIR'])
    else:
        out = Path(os.environ['SKA'], 'data', 'agasc')
    return out


def default_agasc_file():
    """Default main AGASC file ``agasc_dir() / miniagasc.h5``.

    :returns: str
    """
    return str(default_agasc_dir() / 'miniagasc.h5')


@lru_cache_timed(timeout=3600)
def get_supplement_table(name, agasc_dir=None, as_dict=False):
    """Get one of the tables in the AGASC supplement.

    This function gets one of the supplement tables, specified with ``name``:

    - ``bad``: Bad stars (agasc_id, source)
    - ``mags``: Estimated mags (agasc_id, mag_aca mag_aca_err)
    - ``obs``: Star-obsid status for mag estimation (agasc_id, obsid, ok,
      comments)

    This function is cached with a timeout of an hour, so you can call it
    repeatedly with no penalty in performance.

    If ``as_dict=False`` (default) then the table is returned as an astropy
    ``Table``.

    If ``as_dict=True`` then the table is returned as a dict of {key: value}
    pairs. For ``mags`` and ``bad``, the key is ``agasc_id``. For ``obs`` the
    key is the ``(agasc_id, obsid)`` tuple. In all cases the value is a dict
    of the remaining columns.

    :param name: Table name within the AGASC supplement HDF5 file
    :param data_root: directory containing the AGASC supplement HDF5 file
        (default=same directory as the AGASC file)
    :param as_dict: return result as a dictionary (default=False)

    :returns: supplement table as ``Table`` or ``dict``
    """
    if agasc_dir is None:
        agasc_dir = default_agasc_dir()

    if name not in ('mags', 'bad', 'obs'):
        raise ValueError("table name must be one of 'mags', 'bad', or 'obs'")

    supplement_file = agasc_dir / 'agasc_supplement.h5'
    with tables.open_file(supplement_file) as h5:
        try:
            dat = getattr(h5.root, name)[:]
        except tables.NoSuchNodeError:
            warnings.warn(f"No dataset '{name}' in {supplement_file},"
                          " returning empty table")
            dat = []

    if as_dict:
        out = {}
        keys_names = {
            'mags': ['agasc_id'],
            'bad': ['agasc_id'],
            'obs': ['agasc_id', 'obsid']}
        key_names = keys_names[name]
        for row in dat:
            # Make the key, coercing the values from numpy to native Python
            key = tuple(row[nm].item() for nm in key_names)
            if len(key) == 1:
                key = key[0]
            # Make the value from the remaining non-key column names
            out[key] = {nm: row[nm].item() for nm in row.dtype.names if nm not in key_names}
    else:
        out = Table(dat)

    return out


def disable_supplement(func):
    """Decorator to temporarily disable use of the AGASC supplement in queries.

    This is mostly for testing or specialized applications to override the
    default behavior to use the AGASC supplement star mags when available.
    """
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        orig = os.environ.get(DISABLE_SUPPLEMENT_ENV)
        os.environ[DISABLE_SUPPLEMENT_ENV] = '1'
        try:
            return func(*args, **kwargs)
        finally:
            if orig is None:
                del os.environ[DISABLE_SUPPLEMENT_ENV]
            else:
                os.environ[DISABLE_SUPPLEMENT_ENV] = orig

    return wrap


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
        # Read the file of RA and DEC values (sorted on DEC):
        #  dec: DEC values
        #  ra: RA values
        with tables.open_file(self.agasc_file) as h5:
            radecs = h5.root.data[:][['RA', 'DEC']]

            # Now copy to separate ndarrays for memory efficiency
            return radecs['RA'].copy(), radecs['DEC'].copy()


def get_ra_decs(agasc_file):
    agasc_file = os.path.abspath(agasc_file)
    if agasc_file not in RA_DECS_CACHE:
        RA_DECS_CACHE[agasc_file] = RaDec(agasc_file)
    return RA_DECS_CACHE[agasc_file]


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
                   pm_filter=True, fix_color1=True, use_supplement=False):
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
    :param use_supplement: Use estimated mag from AGASC supplement where available (default=False)

    :returns: astropy Table of AGASC entries
    """
    if agasc_file is None:
        agasc_file = default_agasc_file()

    # Possibly expand initial radius to allow for slop due proper motion
    rad_pm = radius + (0.1 if pm_filter else 0.0)

    ra_decs = get_ra_decs(agasc_file)

    idx0, idx1 = np.searchsorted(ra_decs.dec, [dec - rad_pm, dec + rad_pm])

    dists = sphere_dist(ra, dec, ra_decs.ra[idx0:idx1], ra_decs.dec[idx0:idx1])
    ok = dists <= rad_pm

    with tables.open_file(agasc_file) as h5:
        stars = Table(h5.root.data[idx0:idx1][ok], copy=False)

    add_pmcorr_columns(stars, date)
    if fix_color1:
        update_color1_column(stars)

    # Final filtering using proper-motion corrected positions
    if pm_filter:
        dists = sphere_dist(ra, dec, stars['RA_PMCORR'], stars['DEC_PMCORR'])
        ok = dists <= radius
        stars = stars[ok]

    if use_supplement:
        update_from_supplement(stars)

    return stars


def get_star(id, agasc_file=None, date=None, fix_color1=True, use_supplement=False):
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
    :param use_supplement: Use estimated mag from AGASC supplement where available (default=False)
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

    if use_supplement:
        update_from_supplement(t)

    return t[0]


def get_stars(ids, agasc_file=None, dates=None, fix_color1=True, use_supplement=False):
    """
    Get AGASC catalog entries for star ``ids`` at ``dates``.

    The input ``ids`` and ``dates`` are broadcast together for the output shape
    (though note that the result is flattened in the end). If both are scalar
    inputs then the output is a Table Row, otherwise the output is a Table.

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
    :param use_supplement: Use estimated mag from AGASC supplement where available (default=False)
    :returns: astropy Table of AGASC entries, or Table Row of one entry for scalar input
    """

    if agasc_file is None:
        agasc_file = default_agasc_file()

    rows = []
    dates = DateTime(dates).date

    with tables.open_file(agasc_file) as h5:
        tbl = h5.root.data
        ids, dates = np.broadcast_arrays(ids, dates)
        for id, date in zip(np.atleast_1d(ids), np.atleast_1d(dates)):
            id_rows = tbl.read_where('(AGASC_ID == {})'.format(id))

            if len(id_rows) > 1:
                raise InconsistentCatalogError(
                    f'More than one entry found for {id} in AGASC')

            if id_rows is None or len(id_rows) == 0:
                raise IdNotFound(f'No entry found for {id} in AGASC')

            rows.append(id_rows[0])

    t = Table(np.vstack(rows).flatten())

    add_pmcorr_columns(t, dates)
    if fix_color1:
        update_color1_column(t)
    t['DATE'] = dates

    if use_supplement:
        update_from_supplement(t)

    return t if ids.shape else t[0]


# Interpolate COMMON_DOC into those function docstrings
for func in get_stars, get_star, get_agasc_cone:
    func.__doc__ = func.__doc__.format(common_doc=COMMON_DOC)


def update_from_supplement(stars):
    """Overwrite mag and color1 information from AGASC supplement in ``stars``.

    Stars with available mag estimates in   the AGASC supplement are updated
    in-place in the input ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT`` (128).
    - If COLOR1 is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.

    This functionality is gloabally disabled if the environment variable
    ``AGASC_DISABLE_SUPPLEMENT`` is set to any value.

    :param stars: astropy.table.Table of stars
    """
    if DISABLE_SUPPLEMENT_ENV in os.environ:
        return

    def set_star(star, name, value):
        """Set star[name] = value if ``name`` is a column in the table"""
        try:
            star[name] = value
        except KeyError:
            pass

    # Get estimate mags and errs from supplement as a dict of dict
    # agasc_id : {mag_aca: .., mag_aca_err: ..}.
    supplement_mags = get_supplement_table('mags', as_dict=True)

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
