# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import numpy as np
import numexpr
import tables

from ska_path import ska_path
from Chandra.Time import DateTime
from astropy.table import Table, Column

__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star']

DATA_ROOT = ska_path('data', 'agasc')

DEFAULT_AGASC_FILE = os.path.join(DATA_ROOT, 'miniagasc.h5')


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

    numerator = numexpr.evaluate('sin((dec2 - dec1) / 2) ** 2 + '
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
    # Compute the multiplicative factor to convert from the AGASC proper motion
    # field to degrees.  The AGASC PM is specified in milliarcsecs / year, so this
    # is dyear * (degrees / milliarcsec)

    # The dyear for proper motion is only relevant for stars that have defined proper motion
    # so set to zero for all stars by default
    dyear = np.zeros(len(stars))
    has_pm = (stars['PM_DEC'] != -9999) | (stars['PM_RA'] != -9999)
    # For most of them, the epoch is fixed at 2000, so we don't need N calls to DateTime to
    # figure that out
    epoch_is_2000 = (stars['EPOCH'] == 2000.0)
    ok = has_pm & epoch_is_2000
    dyear[ok] = (DateTime(date) - DateTime(2000, format='frac_year')) / 365.25
    # For stars with proper motion correction but epoch != 2000, calculate individually.
    ok = has_pm & ~epoch_is_2000
    dyear[ok] = (DateTime(date) - DateTime(stars['EPOCH'][ok], format='frac_year')) / 365.25
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
                   pm_filter=True, fix_color1=True):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    The star positions are corrected for proper motion using the ``date``
    argument, which defaults to the current date if not supplied.  The
    corrected positions are available in the ``RA_PMCORR`` and ``DEC_PMCORR``
    columns, respectively.

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

    return stars


def get_star(id, agasc_file=None, date=None, fix_color1=True):
    """
    Get AGASC catalog entry for star with requested id.

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
    :returns: astropy Table Row of entry for id
    """

    if agasc_file is None:
        agasc_file = DEFAULT_AGASC_FILE

    with tables_open_file(agasc_file) as h5:
        tbl = h5.root.data
        tbl_read_where = getattr(tbl, 'read_where', None) or tbl.readWhere
        id_rows = tbl_read_where('(AGASC_ID == {})'.format(id))

    if len(id_rows) > 1:
        raise InconsistentCatalogError(
            "More than one entry found for {} in AGASC".format(id))

    if id_rows is None or len(id_rows) == 0:
        raise IdNotFound()

    t = Table(id_rows)
    add_pmcorr_columns(t, date)
    if fix_color1:
        update_color1_column(t)

    return t[0]
