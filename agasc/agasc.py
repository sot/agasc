import os

import numpy as np
import numexpr
import tables

from ska_path import ska_path
from Chandra.Time import DateTime
from astropy.table import Table, Column

__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star']

DATA_ROOT = ska_path('data', 'agasc')


class IdNotFound(LookupError):
    pass


class InconsistentCatalogError(Exception):
    pass


class RaDec(object):
    def __init__(self):
        pass

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
        radecs = np.load(os.path.join(DATA_ROOT, 'ra_dec.npy'))

        # Now copy to separate ndarrays for memory efficiency
        return radecs['ra'].copy(), radecs['dec'].copy()

RA_DECS = RaDec()


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

    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)

    numerator = numexpr.evaluate('sin((dec2 - dec1) / 2) ** 2 + '
                                 'cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2')

    dists = numexpr.evaluate('2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)')
    return np.degrees(dists)


def add_pmcorr_columns(stars, date):
    # Compute the multiplicative factor to convert from the AGASC proper motion
    # field to degrees.  The AGASC PM is specified in milliarcsecs / year, so this
    # is dyear * (degrees / milliarcsec)
    agasc_equinox = DateTime('2000:001:00:00:00.000')
    dyear = (DateTime(date) - agasc_equinox) / 365.25
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


def get_agasc_cone(ra, dec, radius=1.5, date=None, agasc_file=None):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    The star positions are corrected for proper motion using the ``date``
    argument, which defaults to the current date if not supplied.  The
    corrected positions are available in the ``RA_PMCORR`` and ``DEC_PMCORR``
    columns, respectively.

    The default ``agasc_file`` is ``/proj/sot/ska/data/agasc/miniagasc.h5``

    Example::

      >>> import agasc
      >>> stars = agasc.get_agasc_cone(10.0, 20.0, 1.5)
      >>> plt.plot(stars['RA'], stars['DEC'], '.')

    :param ra: RA (deg)
    :param dec: Declination (deg)
    :param radius: Cone search radius (deg)
    :param date: Date for proper motion (default=Now)
    :param agasc_file: Mini-agasc HDF5 file sorted by Dec (optional)

    :returns: astropy Table of AGASC entries
    """
    if agasc_file is None:
        agasc_file = os.path.join(DATA_ROOT, 'miniagasc.h5')

    idx0, idx1 = np.searchsorted(RA_DECS.dec, [dec - radius, dec + radius])

    dists = sphere_dist(ra, dec, RA_DECS.ra[idx0:idx1], RA_DECS.dec[idx0:idx1])
    ok = dists <= radius

    with tables.openFile(agasc_file) as h5:
        stars = Table(h5.root.data[idx0:idx1][ok], copy=False)

    add_pmcorr_columns(stars, date)

    return stars


def get_star(id, agasc_file=None, date=None):
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
    :returns: astropy Table Row of entry for id
    """

    if agasc_file is None:
        agasc_file = os.path.join(DATA_ROOT, 'miniagasc.h5')

    with tables.openFile(agasc_file) as h5:
        tbl = h5.root.data
        id_rows = tbl.readWhere('(AGASC_ID == {})'.format(id))

    if len(id_rows) > 1:
        raise InconsistentCatalogError(
            "More than one entry found for {} in AGASC".format(id))

    if id_rows is None or len(id_rows) == 0:
        raise IdNotFound()

    t = Table(id_rows)
    add_pmcorr_columns(t, date)

    return t[0]
