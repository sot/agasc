import os

import numpy as np
import numexpr
import tables

from Chandra.Time import DateTime
from astropy.table import Table, Column

__all__ = ['sphere_dist', 'get_agasc_cone', 'get_star']


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
        radecs = np.load(os.path.join(os.path.dirname(__file__), 'ra_dec.npy'))

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

    :returns: table of AGASC entries
    """
    if agasc_file is None:
        agasc_file = os.path.join('/', 'proj', 'sot', 'ska', 'data', 'agasc', 'miniagasc.h5')

    idx0, idx1 = np.searchsorted(RA_DECS.dec, [dec - radius, dec + radius])

    dists = sphere_dist(ra, dec, RA_DECS.ra[idx0:idx1], RA_DECS.dec[idx0:idx1])
    ok = dists <= radius

    h5 = tables.openFile(agasc_file)
    stars = h5.root.data[idx0:idx1][ok]
    h5.close()

    # Compute the multiplicative factor to convert from the AGASC proper motion
    # field to degrees.  The AGASC PM is specified in milliarcsecs / year, so this
    # is dyear * (degrees / milliarcsec)
    agasc_equinox = DateTime('2000:001:00:00:00.000')
    dyear = (DateTime(date) - agasc_equinox) / 365.25
    pm_to_degrees = dyear / (3600. * 1000.)

    pm_corr = {}  # Proper-motion corrected coordinates
    for axis, axis_PM in (('RA', 'PM_RA'),
                          ('DEC', 'PM_DEC')):
        pm_corr[axis] = stars[axis].copy()
        ok = stars[axis_PM] != -9999  # Select stars with an available PM correction
        pm_corr[axis][ok] = stars[axis][ok] + stars[axis_PM][ok] * pm_to_degrees

    # Add the proper-motion corrected columns to table using astropy.table.Table
    stars = Table(stars, copy=False)
    stars.add_columns([Column(data=pm_corr['RA'], name='RA_PMCORR'),
                       Column(data=pm_corr['DEC'], name='DEC_PMCORR')])

    return stars

def get_star(id, agasc_file=None):
    """
    Get AGASC catalog entry for star with requested id.

    :param id: AGASC id
    :returns: record of entry for id (or None if not found)
    """

    if agasc_file is None:
        agasc_file = os.path.join('/', 'proj', 'sot', 'ska', 'data', 'agasc', 'miniagasc.h5')

    h5 = tables.openFile(agasc_file)
    tbl = h5.root.data
    id_rows = tbl.readWhere('(AGASC_ID == {})'.format(id))
    h5.close()

    if len(id_rows) > 1:
        return ValueError("More than one entry found for {} in AGASC".format(id))
    if id_rows is None or len(id_rows) == 0:
        return None
    return id_rows[0]
