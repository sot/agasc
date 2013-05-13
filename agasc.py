import os
import tables
from Chandra.Time import DateTime
import Ska.Numpy


def agasc(ra, dec, radius=1.5, date=None,
          pm_correct=True, agasc_file=None):
    if agasc_file is None:
        agasc_file = os.path.join(os.environ['SKA_DATA'],
                                  'agasc',
                                  'miniagasc.h5')
    if date is None:
        date = DateTime()

    # determine ranges for "box" search of RA and Dec
    ras = []
    ra_min = ra - radius
    ra_max = ra + radius
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
    decs = []
    dec_min = dec - radius
    dec_max = dec + radius
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
