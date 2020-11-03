# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

import numpy as np
import numexpr
import tables
import warnings

from Chandra.Time import DateTime
from astropy.table import Table, Column
from ska_helpers.utils import LazyDict


__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star', 'get_stars', 'MAG_CATID_SUPPLEMENT']

DATA_ROOT = Path(os.environ['SKA'], 'data', 'agasc')
DEFAULT_AGASC_FILE = str(DATA_ROOT / 'miniagasc.h5')
DEFAULT_SUPPLEMENT_FILE = str(DATA_ROOT / 'agasc_supplement.h5')

MAG_CATID_SUPPLEMENT = 128


def load_supplement():
    supplement = {}
    with tables.open_file(DEFAULT_SUPPLEMENT_FILE) as h5:
        try:
            stars = h5.root.mags
        except tables.NoSuchNodeError:
            warnings.warn('No observed star magnitude data in agasc_supplement.h5')
        else:
            for star in stars:
                # Use Python int for key not numpy.int32
                supplement[int(star['agasc_id'])] = (star['mag_aca'], star['mag_aca_err'])

    return supplement


SUPPLEMENT = LazyDict(load_supplement)


class IdNotFound(LookupError):
    pass


class InconsistentCatalogError(Exception):
    pass


def tables_open_file(filename):
    """
    Open an HDF5 file using table, but allow for a Path object input.

    :param filename: table file name (str, Path)
    :returns: h5 handle
    """
    return tables.open_file(str(filename))


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
        with tables_open_file(self.agasc_file) as h5:
            radecs = h5.root.data[:][['RA', 'DEC']]

            # Now copy to separate ndarrays for memory efficiency
            return radecs['RA'].copy(), radecs['DEC'].copy()


RA_DECS_CACHE = {DEFAULT_AGASC_FILE: RaDec(DEFAULT_AGASC_FILE)}


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
                   pm_filter=True, fix_color1=True, use_mag_est=False):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    The star positions are corrected for proper motion using the ``date``
    argument, which defaults to the current date if not supplied.  The
    corrected positions are available in the ``RA_PMCORR`` and ``DEC_PMCORR``
    columns, respectively.

    If ``use_mag_est`` is ``True``, then stars with available mag estimates in
    the AGASC supplement are updated in-place in the output ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT`` (128).
    - If COLOR1 is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.

    The default ``agasc_file`` is ``$SKA/data/agasc/miniagasc.h5``

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
    :param use_mag_est: Use estimated mag from AGASC supplement where available (default=False)

    :returns: astropy Table of AGASC entries
    """
    if agasc_file is None:
        agasc_file = DEFAULT_AGASC_FILE

    # Possibly expand initial radius to allow for slop due proper motion
    rad_pm = radius + (0.1 if pm_filter else 0.0)

    ra_decs = get_ra_decs(agasc_file)

    idx0, idx1 = np.searchsorted(ra_decs.dec, [dec - rad_pm, dec + rad_pm])

    dists = sphere_dist(ra, dec, ra_decs.ra[idx0:idx1], ra_decs.dec[idx0:idx1])
    ok = dists <= rad_pm

    with tables_open_file(agasc_file) as h5:
        stars = Table(h5.root.data[idx0:idx1][ok], copy=False)

    add_pmcorr_columns(stars, date)
    if fix_color1:
        update_color1_column(stars)

    # Final filtering using proper-motion corrected positions
    if pm_filter:
        dists = sphere_dist(ra, dec, stars['RA_PMCORR'], stars['DEC_PMCORR'])
        ok = dists <= radius
        stars = stars[ok]

    if use_mag_est:
        update_mags_from_supplement(stars)

    return stars


def get_star(id, agasc_file=None, date=None, fix_color1=True, use_mag_est=False):
    """
    Get AGASC catalog entry for star with requested id.

    If ``use_mag_est`` is ``True``, then stars with available mag estimates in
    the AGASC supplement are updated in-place in the output star entry.

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT`` (128).
    - If COLOR1 is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.

    The default ``agasc_file`` is ``/proj/sot/ska/data/agasc/miniagasc.h5``

    Example::

      >>> import agasc
      >>> star = agasc.get_star(636629880)
      >>> for name in star.colnames:
      ...     print '{:12s} : {}'.format(name, star[name])
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
    :param use_mag_est: Use estimated mag from AGASC supplement where available (default=False)
    :returns: astropy Table Row of entry for id
    """

    if agasc_file is None:
        agasc_file = DEFAULT_AGASC_FILE

    with tables_open_file(agasc_file) as h5:
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

    if use_mag_est:
        update_mags_from_supplement(t)

    return t[0]


def get_stars(ids, agasc_file=None, dates=None, fix_color1=True, use_mag_est=False):
    """
    Get AGASC catalog entries for star ``ids`` at ``dates``.

    The input ``ids`` and ``dates`` are broadcast together for the output shape
    (though note that the result is flattened in the end). If both are scalar
    inputs then the output is a Table Row, otherwise the output is a Table.

    Unlike the similar ``get_star`` function, this adds a ``DATE`` column
    indicating the date at which the star coordinates (RA_PMCORR, DEC_PMCORR)
    are computed.

    If ``use_mag_est`` is ``True``, then stars with available mag estimates in
    the AGASC supplement are updated in-place in the output ``stars`` Table:

    - ``MAG_ACA`` and ``MAG_ACA_ERR`` are set according to the supplement.
    - ``MAG_CATID`` (mag catalog ID) is set to ``MAG_CATID_SUPPLEMENT`` (128).
    - If COLOR1 is 0.7 or 1.5 then it is changed to 0.69 or 1.49 respectively.

    The default ``agasc_file`` is ``$SKA/data/agasc/miniagasc.h5``

    Example::
      >>> import agasc
      >>> star = agasc.get_stars(636629880)
      >>> for name in star.colnames:
      ...     print '{:12s} : {}'.format(name, star[name])
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
    :param use_mag_est: Use estimated mag from AGASC supplement where available (default=False)
    :returns: astropy Table of AGASC entries, or Table Row of one entry for scalar input
    """

    if agasc_file is None:
        agasc_file = DEFAULT_AGASC_FILE

    rows = []
    dates = DateTime(dates).date

    with tables_open_file(agasc_file) as h5:
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

    if use_mag_est:
        update_mags_from_supplement(t)

    return t if ids.shape else t[0]


def update_mags_from_supplement(stars):
    """Overwrite mag and color1 information from AGASC supplement in ``stars``.

    Stars with available mag estimates in the supplement are updated in-place in
    the ``stars`` Table. The catalog ID is set to ``MAG_CATID_SUPPLEMENT`` (128).
    Where COLOR1 is 0.7 or 1.5 it is changed to 0.69 or 1.49 respectively.

    :param stars: astropy.table.Table of stars
    """
    for idx, agasc_id in enumerate(stars['AGASC_ID']):
        agasc_id = int(agasc_id)
        if agasc_id in SUPPLEMENT:
            mag_est, mag_est_err = SUPPLEMENT[agasc_id]
            stars['MAG_ACA'][idx] = mag_est
            # Mag err is stored as int16 in units of 0.01 mag. Use same convention here.
            stars['MAG_ACA_ERR'][idx] = round(mag_est_err * 100)
            stars['MAG_CATID'][idx] = MAG_CATID_SUPPLEMENT
            color1 = stars['COLOR1'][idx]
            if np.isclose(color1, 0.7) or np.isclose(color1, 1.5):
                stars['COLOR1'][idx] = color1 - 0.01
