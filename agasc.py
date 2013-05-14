import os

import numpy as np
import numexpr
import tables

from Chandra.Time import DateTime
from astropy.table import Table, Column

# Read the file of RA and DEC indexes:
#  dec: sorted DEC values
#  dec_idx : corresponding indexes into miniagasc
#  ra: sorted RA values
#  ra_idx: corresponding indexes into miniagasc
RDI = np.load(os.path.join(os.path.dirname(__file__), 'ra_dec.npy'))

# Now copy to separate ndarrays for memory efficiency
RA = RDI['ra'].copy()
DEC = RDI['dec'].copy()
del RDI


def sphere_dist(lon1, lat1, lon2, lat2):
    """
    Haversine formula for angular distance on a sphere: more stable at poles.
    This version uses arctan instead of arcsin and thus does better with sign
    conventions.

    Inputs must be in degrees.  Output is in degrees.
    """

    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    numerator = numexpr.evaluate('sin((lat2 - lat1) / 2) ** 2 + '
                                 'cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2')

    dists = numexpr.evaluate('2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)')
    return np.degrees(dists)


def get_agasc_cone(ra, dec, radius=1.5, date=None, agasc_file=None):
    """
    Get AGASC stars within ``radius`` degrees of ``ra``, ``dec``.

    :param ra: RA (deg)
    :param dec: Declination (deg)
    :param radius: Cone search radius (deg)
    :param date: Date for proper motion (default=Now)
    :agasc_file: Mini-agasc HDF5 file sorted by Dec (optional)

    :returns: table of AGASC stars
    """
    if agasc_file is None:
        agasc_file = os.path.join(os.environ['SKA_DATA'], 'agasc', 'miniagasc.h5')

    dec0 = dec - radius
    dec1 = dec + radius
    idx0, idx1 = np.searchsorted(DEC, [dec0, dec1])

    dists = sphere_dist(ra, dec, RA[idx0:idx1], DEC[idx0:idx1])
    ok = dists < radius
    stars_idxs = np.arange(idx0, idx1)[ok]

    h5 = tables.openFile(agasc_file)
    stars = h5.root.data.readCoordinates(stars_idxs)
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
        ok = pm_corr[axis] != -9999  # Select stars with an available PM correction
        pm_corr[axis][ok] = stars[axis][ok] + stars[axis_PM][ok] * pm_to_degrees

    # Add the proper-motion corrected columns to table using astropy.table.Table
    stars = Table(stars, copy=False)
    stars.add_columns([Column(data=pm_corr['RA'], name='RA_PMCORR'),
                       Column(data=pm_corr['DEC'], name='DEC_PMCORR')])

    return stars


def agasc(ra, dec, radius=1.5, date=None,
          pm_correct=True, agasc_file=None):
    import Ska.Numpy
    if agasc_file is None:
        agasc_file = os.path.join(os.environ['SKA_DATA'],
                                  'agasc',
                                  'miniagasc.h5')
    if date is None:
        date = DateTime()

    # determine ranges for "box" search of RA and Dec
    decs = []
    dec_min = dec - radius
    dec_max = dec + radius
    max_abs_dec = max(abs(dec_min), abs(dec_max))
    if dec_min < -90:
        decs.append([180 + dec_min, 90],
                      [-90, dec])
    else:
        decs.append([dec_min, dec])
    if dec_max > 90:
        decs.append([dec, 90],
                      [-90, dec_max - 180])
    else:
        decs.append([dec, dec_max])

    ras = []
    ra_min = ra - radius / np.cos(np.radians(max_abs_dec))
    ra_max = ra + radius / np.cos(np.radians(max_abs_dec))
    if ra_min < 0:
        ras.append([360 + ra_min, 360])
        ras.append([0, ra])
    else:
        ras.append([ra_min, ra])
    if ra_max > 360:
        ras.append([ra, 360])
        ras.append([0, ra_max - 360])
    else:
        ras.append([ra, ra_max])

    query = ("("
             + " | ".join(["((RA >= %f) & (RA <= %f)) " % (ra_r[0], ra_r[1])
                         for ra_r in ras])
             + ") & (" 
             + " | ".join([" ((DEC >= %f) & (DEC <= %f)) " % (dec_r[0], dec_r[1])
                           for dec_r in decs])
             + ")")

    h5 = tables.openFile(agasc_file)
    tbl = h5.getNode('/', 'data')
    get_coord_match = tbl.getWhereList(query)
    table = tbl.readCoordinates(get_coord_match)
    h5.close()
    if not pm_correct:
        return table

    agasc_start_date = DateTime('2000:001:00:00:00.000')
    dsecs = date.secs - agasc_start_date.secs
    dyear = dsecs / (86400 * 365.25)
    milliarcsecs_per_degree = 3600 * 1000

    ra_corr = table['RA'].copy()
    has_ra_pm = table['PM_RA'] != -9999
    ra_corr[has_ra_pm] = (table[has_ra_pm]['RA']
                          + (table[has_ra_pm]['PM_RA']
                             * (dyear / milliarcsecs_per_degree)))

    dec_corr = table['DEC'].copy()
    has_dec_pm = table['PM_DEC'] != -9999
    dec_corr[has_dec_pm] = (table[has_dec_pm]['DEC']
                            + (table[has_dec_pm]['PM_DEC']
                                * (dyear / milliarcsecs_per_degree)))

    add_ra = Ska.Numpy.add_column(table, 'RA_PMCORR', ra_corr)
    corr = Ska.Numpy.add_column(add_ra, 'DEC_PMCORR', dec_corr)

    return corr
