"""
Functions to estimate observed ACA magnitudes
"""

import sys
import traceback
import logging
import collections

import scipy.stats
import scipy.special
import numpy as np
import numba
from astropy.table import Table, vstack

from Chandra.Time import DateTime
from cheta import fetch
from Quaternion import Quat
import Ska.quatutil
from mica.archive import aca_l0
from mica.archive.aca_dark.dark_cal import get_dark_cal_image
from chandra_aca.transform import count_rate_to_mag, pixels_to_yagzag
from cxotime import CxoTime
from kadi import events

from . import star_obs_catalogs
from agasc import get_star


logger = logging.getLogger('agasc.supplement')


MAX_MAG = 15
MASK = {
    'mouse_bit': np.array([[True, True, True, True, True, True, True, True],
                           [True, True, False, False, False, False, True, True],
                           [True, False, False, False, False, False, False, True],
                           [True, False, False, False, False, False, False, True],
                           [True, False, False, False, False, False, False, True],
                           [True, False, False, False, False, False, False, True],
                           [True, True, False, False, False, False, True, True],
                           [True, True, True, True, True, True, True, True]])
}


EXCEPTION_MSG = {
    -1: 'Unknown',
    0: 'OK',
    1: 'No level 0 data',
    2: 'No telemetry data',
    3: 'Mismatch in telemetry between aca_l0 and cheta',
    4: 'Time mismatch between cheta and level0',
    5: 'Failed job',
    6: 'Suspect observation'
}
EXCEPTION_CODES = collections.defaultdict(lambda: -1)
EXCEPTION_CODES.update({msg: code for code, msg in EXCEPTION_MSG.items() if code > 0})


class MagStatsException(Exception):
    def __init__(self, msg='', agasc_id=None, obsid=None, timeline_id=None, mp_starcat_time=None,
                 **kwargs):
        super().__init__(msg)
        self.error_code = EXCEPTION_CODES[msg]
        self.msg = msg
        self.agasc_id = agasc_id
        self.obsid = obsid[0] if type(obsid) is list and len(obsid) == 1 else obsid
        self.timeline_id = timeline_id
        self.mp_starcat_time = (mp_starcat_time[0] if type(mp_starcat_time) is list
                                and len(mp_starcat_time) == 1 else mp_starcat_time)
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __str__(self):
        return f'MagStatsException: {self.msg} (agasc_id: {self.agasc_id}, ' \
               f'obsid: {self.obsid}, mp_starcat_time: {self.mp_starcat_time})'

    def __iter__(self):
        yield 'error_code', self.error_code
        yield 'msg', self.msg
        yield 'agasc_id', self.agasc_id
        yield 'obsid', self.obsid
        yield 'timeline_id', self.timeline_id
        yield 'mp_starcat_time', self.mp_starcat_time


def _magnitude_correction(time, mag_aca):
    """
    Get a time-dependent correction to AOACMAG (prior to dynamic background subtraction).

    :param time: Chandra.Time.DateTime
    :param mag_aca: np.array
    :return: np.array
    """
    params = {"t_ref": "2011-01-01 12:00:00.000",
              "p": [0.005899340720522751,
                    0.12029019332761458,
                    -2.99386247406073e-10,
                    -6.9534637950633265,
                    0.7916261423307238]}

    q = params['p']
    t_ref = DateTime(params['t_ref'])
    dmag = (q[0] + (q[1] + q[2] * np.atleast_1d(time))
            * np.exp(q[3] + q[4] * np.atleast_1d(mag_aca)))
    dmag[np.atleast_1d(time) < t_ref.secs] = 0
    return np.squeeze(dmag)


def get_responsivity(time):
    """
    ACA magnitude response over time.

    This was estimated with bright stars that were observed more than a hundred times during the
    mission. More details in the `responsivity notebook`_:

    .. _responsivity notebook: https://nbviewer.jupyter.org/urls/cxc.cfa.harvard.edu/mta/ASPECT/jgonzalez/mag_stats/notebooks/03-high_mag_responsivity-fit.ipynb  # noqa

    :param time: float
        Time in CXC seconds
    :return:
    """
    a, b, c = [3.19776750e-02, 5.35201479e+08, 8.49670756e+07]
    return - a * (1 + scipy.special.erf((time - b) / c)) / 2


def get_droop_systematic_shift(magnitude):
    """
    Difference between the magnitude determined from DC-subtracted image telemetry and
    the catalog ACA magnitude.

    The magnitude shift is time-independent. It depends only on the catalog magnitude and is zero
    for bright stars. More details in the `droop notebook`_:

    .. _droop notebook: https://nbviewer.jupyter.org/urls/cxc.cfa.harvard.edu/mta/ASPECT/jgonzalez/mag_stats/notebooks/04-DroopAfterSubtractionAndResponsivity-fit.ipynb  # noqa

    :param magnitude: float
        Catalog ACA magnitude
    :return:
    """
    a, b = [11.25572, 0.59486369]
    return np.exp((magnitude - a) / b)


def rolling_mean(t, f, window, selection=None):
    """
    Calculate the rolling mean of the 'f' array, using a centered square window in time.

    :param t: np.array
        the time array.
    :param f: np.array
        the array to average.
    :param window: float
        the window size (in the same units as the time array).
    :param selection:  np.array
        An optional array of bool.
    :return: np.array
        An array with the same type and shape as 'f'
    """
    result = np.ones_like(f) * np.nan
    if selection is None:
        selection = np.ones_like(f, dtype=bool)

    assert len(f) == len(t)
    assert len(f) == len(selection)
    assert len(selection.shape) == 1

    _rolling_mean_(result, t, f, window, selection)
    return result


@numba.jit(nopython=True)
def _rolling_mean_(result, t, f, window, selection):
    i_min = 0
    i_max = 0
    n = 0
    f_sum = 0
    for i in range(len(f)):
        if not selection[i]:
            continue

        while i_max < len(f) and t[i_max] < t[i] + window / 2:
            if selection[i_max]:
                f_sum += f[i_max]
                n += 1
            i_max += 1
        while t[i_min] < t[i] - window / 2:
            if selection[i_min]:
                f_sum -= f[i_min]
                n -= 1
            i_min += 1
        result[i] = f_sum / n


def get_star_position(star, telem):
    """
    Residuals for a given AGASC record at a given slot/time.

    :param star:
        Table Row of one AGASC entry
    :param telem: table
        Table with columns AOATTQT1, AOATTQT2, AOATTQT3, AOATTQT4.
    :return:
    """
    aca_misalign = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rad_to_arcsec = 206264.81

    q = np.array([telem['AOATTQT1'],
                  telem['AOATTQT2'],
                  telem['AOATTQT3'],
                  telem['AOATTQT4']]).transpose()
    norm = np.sum(q**2, axis=1, keepdims=True)
    # I am just normalizing q, just in case.
    n = np.squeeze(np.sqrt(norm))
    q[n != 0] /= np.sqrt(norm)[n != 0]  # prevent warning when dividing by zero (it happens)
    q_att = Quat(q=q)
    ts = q_att.transform

    star_pos_eci = Ska.quatutil.radec2eci(star['RA_PMCORR'], star['DEC_PMCORR'])
    d_aca = np.dot(np.dot(aca_misalign, ts.transpose(0, 2, 1)),
                   star_pos_eci).transpose()
    yag = np.arctan2(d_aca[:, 1], d_aca[:, 0]) * rad_to_arcsec
    zag = np.arctan2(d_aca[:, 2], d_aca[:, 0]) * rad_to_arcsec

    logger.debug(f'    star position. AGASC_ID={star["AGASC_ID"]}, '
                 f'{len(yag)} samples, ({yag[0]}, {zag[0]})...')
    return {
        'yang_star': yag,
        'zang_star': zag,
    }


# this is in case one has to return empty telemetry
_telem_dtype = [('times', 'float64'),
                ('IMGSIZE', 'int32'),
                ('IMGROW0', 'int16'),
                ('IMGCOL0', 'int16'),
                ('IMGRAW', 'float32'),
                ('AOACASEQ', '<U4'),
                ('AOPCADMD', '<U4'),
                ('AOATTQT1', 'float64'),
                ('AOATTQT2', 'float64'),
                ('AOATTQT3', 'float64'),
                ('AOATTQT4', 'float64'),
                ('AOACIIR', '<U3'),
                ('AOACISP', '<U3'),
                ('AOACYAN', 'float64'),
                ('AOACZAN', 'float64'),
                ('AOACMAG', 'float32'),
                ('AOACFCT', '<U4'),
                ('mags_img', 'float64'),
                ('yang_img', 'float64'),
                ('zang_img', 'float64'),
                ('yang_star', 'float64'),
                ('zang_star', 'float64'),
                ('mags', 'float64'),
                ('dy', 'float64'),
                ('dz', 'float64'),
                ('dr', 'float64')]


def get_telemetry(obs):
    """
    Get all telemetry relevant for the magnitude estimation task.

    This gets:
    - AOACASEQ
    - AOPCADMD
    - AOACMAG (ACA estimated magnitude)
    - AOACIIR (ACA ionizing radiation flag)
    - AOACISP (ACA saturated pixel flag)

    MSIDs are renamed to remove the slot number.
    This assumes all MSIDs occur at the same times (they do)

    :param obs: astropy.table.Row
        It must have the following columns: 'agasc_id', 'mp_starcat_time', 'mag', 'slot'
    :return: dict
    """
    star_obs_catalogs.load()
    dwell = star_obs_catalogs.DWELLS_NP[star_obs_catalogs.DWELLS_MAP[obs['mp_starcat_time']]]
    star = get_star(obs['agasc_id'], date=dwell['tstart'], use_supplement=False)
    start = dwell['tstart']
    stop = dwell['tstop']
    slot = obs['slot']
    logger.debug(f'  Getting telemetry for AGASC ID={obs["agasc_id"]}, OBSID={obs["obsid"]}, '
                 f'mp_starcat_time={obs["mp_starcat_time"]}')

    # first we get slot data from mica and magnitudes from cheta and match them in time
    # to match them in time, we assume they come in steps of 1.025 seconds, starting from the first
    # time sample.
    slot_data_cols = ['TIME', 'END_INTEG_TIME', 'IMGSIZE',
                      'IMGROW0', 'IMGCOL0', 'TEMPCCD', 'IMGRAW']
    slot_data = aca_l0.get_slot_data(start, stop, slot=obs['slot'],
                                     centered_8x8=True, columns=slot_data_cols)

    names = ['AOACASEQ', 'AOPCADMD', 'CVCMJCTR', 'CVCMNCTR',
             f'AOACIIR{slot}', f'AOACISP{slot}', f'AOACMAG{slot}', f'AOACFCT{slot}',
             f'AOACZAN{slot}', f'AOACYAN{slot}'] + [f'AOATTQT{i}' for i in range(1, 5)]
    msids = fetch.Msidset(names, start, stop)
    if len(slot_data) == 0:
        raise MagStatsException('No level 0 data',
                                agasc_id=obs["agasc_id"],
                                obsid=obs["obsid"],
                                mp_starcat_time=obs["mp_starcat_time"],
                                time_range=[start, stop],
                                slot=obs['slot'])
    times = msids[f'AOACMAG{slot}'].times
    tmin = np.min([np.min(slot_data['END_INTEG_TIME']), np.min(times)])
    t1 = np.round((times - tmin) / 1.025)
    t2 = np.round((slot_data['END_INTEG_TIME'].data - tmin) / 1.025)
    _, i1, i2 = np.intersect1d(t1, t2, return_indices=True)

    times = times[i1]
    slot_data = slot_data[i2]

    if len(times) == 0:
        # the intersection was null.
        raise MagStatsException('Either no telemetry or no matching times between cheta and level0',
                                agasc_id=obs["agasc_id"],
                                obsid=obs["obsid"],
                                mp_starcat_time=obs["mp_starcat_time"])

    # Now that we have the times, we get the rest of the MSIDs
    telem = {
        'times': times
    }
    telem.update({k: slot_data[k] for k in slot_data_cols[2:]})
    telem.update({
        name: msids[name].vals[np.in1d(msids[name].times, times)]
        for name in names
    })

    # get the normal sun and safe sun mode intervals, which will be removed
    excluded_ranges = []
    for event in [events.normal_suns, events.safe_suns]:
        excluded_ranges += event.intervals(times[0] - 4, times[-1] + 4)
    excluded_ranges = [(CxoTime(t[0]).cxcsec, CxoTime(t[1]).cxcsec) for t in excluded_ranges]

    if excluded_ranges:
        excluded = np.zeros_like(times, dtype=bool)
        for excluded_range in excluded_ranges:
            excluded |= ((times >= excluded_range[0]) & (times <= excluded_range[1]))
        telem.update({k: telem[k][~excluded] for k in telem})
        slot_data = slot_data[~excluded]

    if len(slot_data) == 0:
        # the intersection was null.
        raise MagStatsException('Nothing left after removing excluded ranges',
                                agasc_id=obs["agasc_id"],
                                obsid=obs["obsid"],
                                mp_starcat_time=obs["mp_starcat_time"])

    for name in ['AOACIIR', 'AOACISP', 'AOACYAN', 'AOACZAN', 'AOACMAG', 'AOACFCT']:
        telem[name] = telem[f'{name}{slot}']
        del telem[f'{name}{slot}']
    for name in ['AOACIIR', 'AOACISP']:
        telem[name] = np.char.rstrip(telem[name])
    ok = (telem['AOACASEQ'] == 'KALM') & (telem['AOACIIR'] == 'OK') & \
         (telem['AOPCADMD'] == 'NPNT') & (telem['AOACFCT'] == 'TRAK')

    # etc...
    logger.debug('    Adding magnitude estimates')
    telem.update(get_mag_from_img(slot_data, start, ok))
    logger.debug('    Adding star position')
    telem.update(get_star_position(star=star, telem=telem))

    logger.debug('    Correcting for droop')
    droop_shift = get_droop_systematic_shift(star['MAG_ACA'])

    logger.debug('    Correcting for responsivity')
    responsivity = get_responsivity(start)
    telem['mags'] = telem['mags_img'] - responsivity - droop_shift
    telem['mags'][~ok] = 0.
    telem['ok'] = ok

    telem['dy'] = np.ones(len(ok)) * np.inf
    telem['dz'] = np.ones(len(ok)) * np.inf
    telem['dr'] = np.ones(len(ok)) * np.inf
    yang = telem['yang_img'] - telem['yang_star']
    zang = telem['zang_img'] - telem['zang_star']
    rang = np.sqrt(yang**2 + zang**2)
    if np.any(ok & (rang < 10)):
        y25, y50, y75 = np.quantile(yang[ok & (rang < 10)], [0.25, 0.5, 0.75])
        z25, z50, z75 = np.quantile(zang[ok & (rang < 10)], [0.25, 0.5, 0.75])
        centroid_outlier = ((yang > y75 + 3 * (y75 - y25))
                            | (yang < y25 - 3 * (y75 - y25))
                            | (zang > z75 + 3 * (z75 - z25))
                            | (zang < z25 - 3 * (z75 - z25)))

        telem['dy'] = yang - np.mean(yang[ok & ~centroid_outlier])
        telem['dz'] = zang - np.mean(zang[ok & ~centroid_outlier])
        telem['dr'] = (telem['dy'] ** 2 + telem['dz'] ** 2) ** .5

    return telem


def get_telemetry_by_agasc_id(agasc_id, obsid=None, ignore_exceptions=False):
    """
    Get all telemetry relevant for the magnitude estimation, given an AGASC ID.

    This gets all observations of a given star, it gets the telemetry for each, and stacks them.

    :param agasc_id: int
    :param obsid: int (optional)
    :param ignore_exceptions: bool
        if True, any exception is ignored. Useful in some cases. Default is False.
    :return: dict
    """
    logger.debug(f'  Getting telemetry for AGASC ID={agasc_id}')
    star_obs_catalogs.load()
    if obsid is None:
        obs = star_obs_catalogs.STARS_OBS[
            (star_obs_catalogs.STARS_OBS['agasc_id'] == agasc_id)]
    else:
        obs = star_obs_catalogs.STARS_OBS[(star_obs_catalogs.STARS_OBS['agasc_id'] == agasc_id)
                                          & (star_obs_catalogs.STARS_OBS['obsid'] == obsid)]
    if len(obs) > 1:
        obs = obs.loc['mp_starcat_time', sorted(obs['mp_starcat_time'])]
    telem = []
    for i, o in enumerate(obs):
        try:
            t = Table(get_telemetry(o))
            t['obsid'] = o['obsid']
            t['agasc_id'] = agasc_id
            telem.append(t)
        except Exception:
            if not ignore_exceptions:
                logger.info(f'{agasc_id=}, obsid={o["obsid"]} failed')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                trace = traceback.extract_tb(exc_traceback)
                logger.info(f'{exc_type.__name__} {exc_value}')
                for step in trace:
                    logger.info(f'  in {step.filename}:{step.lineno}/{step.name}:')
                    logger.info(f'    {step.line}')
                raise
    return vstack(telem)


def add_obs_info(telem, obs_stats):
    """
    Add observation-specific information to a telemetry table (ok flag, and outlier flag).

    This is done as part of get_agasc_id_stats. It is a convenience for writing reports.

    :param telem: list of tables
        One or more telemetry tables (potentially many observations)
    :param obs_stats: table
        The result of calc_obs_stats.
    :return:
    """
    logger.debug('  Adding observation info to telemetry...')
    obs_stats['obs_ok'] = (
        (obs_stats['n'] > 10)
        & (obs_stats['f_track'] > 0.3)
        & (obs_stats['lf_variability_100s'] < 1)
    )
    obs_stats['comments'] = np.zeros(len(obs_stats), dtype='<U80')

    telem = vstack(telem)
    telem['obs_ok'] = True
    telem['obs_outlier'] = False

    for s in obs_stats:
        obsid = s['obsid']
        o = (telem['obsid'] == obsid)
        telem['obs_ok'][o] = np.ones(np.sum(o), dtype=bool) * s['obs_ok']
        if (np.any(telem['ok'][o]) and s['f_track'] > 0
                and np.isfinite(s['q75']) and np.isfinite(s['q25'])):
            iqr = s['q75'] - s['q25']
            telem['obs_outlier'][o] = (
                telem[o]['ok'] & (iqr > 0)
                & ((telem[o]['mags'] < s['q25'] - 1.5 * iqr)
                   | (telem[o]['mags'] > s['q75'] + 1.5 * iqr))
            )
        logger.debug(f'  Adding observation info to telemetry {obsid=}')
    return telem


@numba.jit(nopython=True)
def staggered_aca_slice(array_in, array_out, row, col):
    for i in np.arange(len(row)):
        if row[i] + 8 < 1024 and col[i] + 8 < 1024:
            array_out[i] = array_in[row[i]:row[i] + 8, col[i]:col[i] + 8]


def get_mag_from_img(slot_data, t_start, ok=True):
    """
    Vectorized estimate of the magnitude from mica archive image telemetry data.

    :param slot_data: astropy.Table.
        The data returned by mica.archive.aca_l0.get_slot_data
    :param t_start:
        The starting time of the observation (by convention, the starcat time)
    :param ok: np.array.
        An boolean array with the same length as slot_data.
        Only magnitudes for entries with ok=True are calculated. The rest are set to MAX_MAG.
    :return:
    """
    logger.debug('    magnitude from images...')
    dark_cal = get_dark_cal_image(t_start, 'nearest',
                                  t_ccd_ref=np.mean(slot_data['TEMPCCD'] - 273.16),
                                  aca_image=False)

    # all images will be 8x8, with a centered mask, imgrow will always be the one of the 8x8 corner.
    imgrow_8x8 = np.where(slot_data['IMGSIZE'] == 8,
                          slot_data['IMGROW0'],
                          slot_data['IMGROW0'] - 1
                          )
    imgcol_8x8 = np.where(slot_data['IMGSIZE'] == 8,
                          slot_data['IMGCOL0'],
                          slot_data['IMGCOL0'] - 1
                          )

    # subtract closest dark cal
    dark = np.zeros([len(slot_data), 8, 8], dtype=np.float64)
    staggered_aca_slice(dark_cal.astype(float), dark, 512 + imgrow_8x8, 512 + imgcol_8x8)
    img_sub = slot_data['IMGRAW'] - dark * 1.696 / 5
    img_sub.mask |= MASK['mouse_bit']

    # calculate magnitude
    mag = np.ones(len(slot_data)) * MAX_MAG
    counts = np.ma.sum(np.ma.sum(img_sub, axis=1), axis=1)
    m = ok & np.isfinite(counts) & (counts > 0)
    mag[m] = count_rate_to_mag(counts[m] * 5 / 1.7)
    mag[mag > MAX_MAG] = MAX_MAG
    # this extra step is to investigate the background scale
    dark = np.ma.array(dark * 1.696 / 5, mask=img_sub.mask)
    img_raw = np.ma.array(slot_data['IMGRAW'], mask=img_sub.mask)
    dark_count = np.ma.sum(np.ma.sum(dark, axis=1), axis=1)
    img_count = np.ma.sum(np.ma.sum(img_raw, axis=1), axis=1)

    # centroids
    yag = np.zeros(len(slot_data))
    zag = np.zeros(len(slot_data))
    pixel_center = np.arange(8) + 0.5
    projected_image = np.ma.sum(slot_data['IMGRAW'], axis=1)
    col = np.ma.sum(pixel_center * projected_image, axis=1) / np.ma.sum(projected_image, axis=1)
    projected_image = np.ma.sum(slot_data['IMGRAW'], axis=2)
    row = np.ma.sum(pixel_center * projected_image, axis=1) / np.ma.sum(projected_image, axis=1)

    y_pixel = row + imgrow_8x8
    z_pixel = col + imgcol_8x8
    yag[m], zag[m] = pixels_to_yagzag(y_pixel[m], z_pixel[m])
    logger.debug(f'    magnitude from images... {len(mag)} samples: {mag[0]:.2f}...')
    return {
        'mags_img': mag,
        'yang_img': yag,
        'zang_img': zag,
        'counts_img': img_count,
        'counts_dark': dark_count
    }


OBS_STATS_INFO = {
    'agasc_id': 'AGASC ID of the star',
    'obsid': 'OBSID corresponding to the dwell when the observation is made',
    'slot': 'Slot number',
    'type': 'GUI/ACQ/BOT',
    'mp_starcat_time':
        'Timestamp (from kadi.commands) for starcat command '
        'preceding the dwell of an observation.',
    'timeline_id': 'starcat command timeline_id from kadi.commands.get_cmds',
    'tstart': 'Dwell start time from kadi.events.manvrs',
    'tstop': 'Dwell end time from kadi.events.manvrs',
    'mag_correction': 'Overall correction applied to the magnitude estimate',
    'responsivity': 'Responsivity correction applied to the magnitude estimate',
    'droop_shift': 'Droop shift correction applied to the magnitude estimate',
    'mag_aca': 'ACA star magnitude from the AGASC catalog',
    'mag_aca_err': 'ACA star magnitude uncertainty from the AGASC catalog',
    'row':
        'Expected row number, based on star location and yanf/zang from mica.archive.starcheck DB',
    'col':
        'Expected col number, based on star location and yanf/zang from mica.archive.starcheck DB',
    'mag_img': 'Magnitude estimate from image telemetry (uncorrected)',
    'mag_obs': 'Estimated ACA star magnitude',
    'mag_obs_err': 'Estimated ACA star magnitude uncertainty',
    'aoacmag_mean': 'Mean of AOACMAG from telemetry',
    'aoacmag_err': 'Standard deviation of AOACMAG from telemetry',
    'aoacmag_q25': '1st quartile of AOACMAG from telemetry',
    'aoacmag_median': 'Median of AOACMAG from telemetry',
    'aoacmag_q75': '3rd quartile of AOACMAG from telemetry',
    'counts_img': 'Raw counts from image telemetry, summed over the mouse-bit window',
    'counts_dark': 'Expected counts from background, summed over the mouse-bit window',
    'f_kalman':
        'Fraction of all samples where AOACASEQ == "KALM" and AOPCADMD == "NPNT" (n_kalman/n)',
    'f_track':
        'Fraction of kalman samples with AOACIIR == "OK" and AOACFCT == "TRAK" (n_track/n_kalman)',
    'f_dr5': 'Fraction of "track" samples with angle residual less than 5 arcsec (n_dr5/n_track)',
    'f_dr3': 'Fraction of "track" samples with angle residual less than 3 arcsec (n_dr3/n_track)',
    'f_ok': 'Fraction of all samples with (kalman & track & dr5) == True (n_ok/n)',
    'q25': '1st quartile of estimated magnitude',
    'median': 'Median of estimated magnitude',
    'q75': '1st quartile of estimated magnitude',
    'mean': 'Mean of estimated magnitude',
    'mean_err': 'Uncrtainty in the mean of estimated magnitude',
    'std': 'Standard deviation of estimated magnitude',
    'skew': 'Skewness of estimated magnitude',
    'kurt': 'Kurtosis of estimated magnitude',
    't_mean': 'Mean of estimated magnitude after removing outliers',
    't_mean_err': 'Uncertainty in the mean of estimated magnitude after removing outliers',
    't_std': 'Standard deviation of estimated magnitude after removing outliers',
    't_skew': 'Skewness of estimated magnitude after removing outliers',
    't_kurt': 'Kurtosis of estimated magnitude after removing outliers',
    'n': 'Number of samples',
    'n_ok': 'Number of samples with (kalman & track & dr5) == True',
    'outliers': 'Number of outliers (+- 3 IQR)',
    'lf_variability_100s': 'Rolling mean of OK magnitudes with a 100 second window',
    'lf_variability_500s': 'Rolling mean of OK magnitudes with a 500 second window',
    'lf_variability_1000s': 'Rolling mean of OK magnitudes with a 1000 second window',
    'tempccd': 'CCD temperature',
    'dr_star': 'Angle residual',
    'obs_ok': 'Boolean flag: everything OK with this observation',
    'obs_suspect': 'Boolean flag: this observation is "suspect"',
    'obs_fail': 'Boolean flag: a processing error when estimating magnitude for this observation',
    'comments': '',
    'w': 'Weight to be used on a weighted mean (1/std)',
    'mean_corrected': 'Corrected mean used in weighted mean (t_mean + mag_correction)',
    'weighted_mean': 'Mean weighted by inverse of standard deviation (mean/std)',
}


def get_obs_stats(obs, telem=None):
    """
    Get summary magnitude statistics for an observation.

    :param obs: astropy.table.Row
        a "star observation" row. From the join of starcheck catalog and starcat commands
        It must have the following columns: 'agasc_id', 'mp_starcat_time', 'mag', 'slot'
    :param telem: dict
        Dictionary with telemetry (output of get_telemetry)
    :return: dict
        dictionary with stats
    """
    logger.debug(f'  Getting OBS stats for AGASC ID {obs["agasc_id"]},'
                 f' OBSID {obs["agasc_id"]} at {obs["mp_starcat_time"]}')

    star_obs_catalogs.load()

    star = get_star(obs['agasc_id'], use_supplement=False)
    dwell = star_obs_catalogs.DWELLS_NP[star_obs_catalogs.DWELLS_MAP[obs['mp_starcat_time']]]
    start = dwell['tstart']
    stop = dwell['tstop']

    stats = {k: obs[k] for k in
             ['agasc_id', 'obsid', 'slot', 'type', 'mp_starcat_time', 'timeline_id']}
    stats['mp_starcat_time'] = stats['mp_starcat_time']
    droop_shift = get_droop_systematic_shift(star['MAG_ACA'])
    responsivity = get_responsivity(start)
    stats.update({'tstart': start,
                  'tstop': stop,
                  'mag_correction': - responsivity - droop_shift,
                  'responsivity': responsivity,
                  'droop_shift': droop_shift,
                  'mag_aca': star['MAG_ACA'],
                  'mag_aca_err': star['MAG_ACA_ERR'] / 100,
                  'row': obs['row'],
                  'col': obs['col'],
                  })

    # other default values
    stats.update({
        'mag_img': np.inf,
        'mag_obs': np.inf,
        'mag_obs_err': np.inf,
        'aoacmag_mean': np.inf,
        'aoacmag_err': np.inf,
        'aoacmag_q25': np.inf,
        'aoacmag_median': np.inf,
        'aoacmag_q75': np.inf,
        'counts_img': np.inf,
        'counts_dark': np.inf,
        'f_kalman': 0.,
        'f_track': 0.,
        'f_dr5': 0.,
        'f_dr3': 0.,
        'f_ok': 0.,
        'q25': np.inf,
        'median': np.inf,
        'q75': np.inf,
        'mean': np.inf,
        'mean_err': np.inf,
        'std': np.inf,
        'skew': np.inf,
        'kurt': np.inf,
        't_mean': np.inf,
        't_mean_err': np.inf,
        't_std': np.inf,
        't_skew': np.inf,
        't_kurt': np.inf,
        'n': 0,
        'n_ok': 0,
        'outliers': -1,
        'lf_variability_100s': np.inf,
        'lf_variability_500s': np.inf,
        'lf_variability_1000s': np.inf,
        'tempccd': np.nan,
        'dr_star': np.inf,
    })

    if telem is None:
        telem = get_telemetry(obs)

    if len(telem) > 0:
        stats.update(calc_obs_stats(telem))
        logger.debug(f'    slot={stats["slot"]}, f_ok={stats["f_ok"]:.3f}, '
                     f'f_track={stats["f_track"]:.3f}, f_dr3={stats["f_dr3"]:.3f},'
                     f' mag={stats["mag_obs"]:.2f}')
    return stats


def calc_obs_stats(telem):
    """
    Get summary magnitude statistics for an observation.

    :param telem: dict
        Dictionary with telemetry (output of get_telemetry)
    :return: dict
        dictionary with stats
    """
    times = telem['times']

    kalman = (telem['AOACASEQ'] == 'KALM') & (telem['AOPCADMD'] == 'NPNT')
    track = (telem['AOACIIR'] == 'OK') & (telem['AOACFCT'] == 'TRAK')
    dr3 = (telem['dy'] < 3) | (telem['dz'] < 3)
    dr5 = (telem['dy'] < 5) | (telem['dz'] < 5)

    f_kalman = np.sum(kalman) / len(kalman)
    n_kalman = np.sum(kalman)
    f_track = np.sum(kalman & track) / n_kalman if n_kalman else 0
    n_track = np.sum(kalman & track)
    f_3 = (np.sum(kalman & track & dr3) / n_track) if n_track else 0
    f_5 = (np.sum(kalman & track & dr5) / n_track) if n_track else 0

    ok = kalman & track & dr3
    n_ok_3 = np.sum(ok)
    n_ok_5 = np.sum(kalman & track & dr5)

    if np.any(ok):
        yang_mean = np.mean(telem['yang_img'][ok] - telem['yang_star'][ok])
        zang_mean = np.mean(telem['zang_img'][ok] - telem['zang_star'][ok])
        dr_star = np.sqrt(yang_mean**2 + zang_mean**2)
    else:
        dr_star = np.inf

    stats = {
        'f_kalman': f_kalman,
        'f_track': f_track,
        'f_dr5': f_5,
        'f_dr3': f_3,
        'f_ok': n_ok_3 / n_kalman,
        'f_ok_3': n_ok_3 / n_kalman,
        'f_ok_5': n_ok_5 / n_kalman,
        'n': len(telem['AOACMAG']),
        'n_ok': n_ok_3,
        'dr_star': dr_star,
    }
    if stats['n_ok'] < 10:
        return stats

    aoacmag_q25, aoacmag_q50, aoacmag_q75 = np.quantile(telem['AOACMAG'][ok], [0.25, 0.5, 0.75])

    mags = telem['mags']
    q25, q50, q75 = np.quantile(mags[ok], [0.25, 0.5, 0.75])
    iqr = q75 - q25
    outlier = ok & ((mags > q75 + 3 * iqr) | (mags < q25 - 3 * iqr))

    s_100s = rolling_mean(times, mags, window=100, selection=ok & ~outlier)
    s_500s = rolling_mean(times, mags, window=500, selection=ok & ~outlier)
    s_1000s = rolling_mean(times, mags, window=1000, selection=ok & ~outlier)

    s_100s = s_100s[~np.isnan(s_100s)]
    s_500s = s_500s[~np.isnan(s_500s)]
    s_1000s = s_1000s[~np.isnan(s_1000s)]

    stats.update({
        'aoacmag_mean': np.mean(telem['AOACMAG'][ok]),
        'aoacmag_err': np.std(telem['AOACMAG'][ok]),
        'aoacmag_q25': aoacmag_q25,
        'aoacmag_median': aoacmag_q50,
        'aoacmag_q75': aoacmag_q75,
        'q25': q25,
        'median': q50,
        'q75': q75,
        'counts_img': np.mean(telem['counts_img'][ok]),
        'counts_dark': np.mean(telem['counts_dark'][ok]),
        'mean': np.mean(mags[ok]),
        'mean_err': scipy.stats.sem(mags[ok]),
        'std': np.std(mags[ok]),
        'skew': scipy.stats.skew(mags),
        'kurt': scipy.stats.kurtosis(mags),
        't_mean': np.mean(mags[ok & (~outlier)]),
        't_mean_err': scipy.stats.sem(mags[ok & (~outlier)]),
        't_std': np.std(mags[ok & (~outlier)]),
        't_skew': scipy.stats.skew(mags[ok & (~outlier)]),
        't_kurt': scipy.stats.kurtosis(mags[ok & (~outlier)]),
        'outliers': np.sum(outlier),
        'lf_variability_100s': np.max(s_100s) - np.min(s_100s),
        'lf_variability_500s': np.max(s_500s) - np.min(s_500s),
        'lf_variability_1000s': np.max(s_1000s) - np.min(s_1000s),
        'tempccd': np.mean(telem['TEMPCCD'][ok]) - 273.16,
    })

    stats.update({
        'mag_img': np.mean(telem['mags_img'][ok & (~outlier)]),
        'mag_obs': stats['t_mean'],
        'mag_obs_err': stats['t_mean_err']
    })

    return stats


AGASC_ID_STATS_INFO = {
    'last_obs_time': 'CXC seconds corresponding to the last mp_starcat_time for the star',
    'agasc_id': 'AGASC ID of the star',
    'mag_aca': 'ACA star magnitude from the AGASC catalog',
    'mag_aca_err': 'ACA star magnitude uncertainty from the AGASC catalog',
    'mag_obs': 'Estimated ACA star magnitude',
    'mag_obs_err': 'Estimated ACA star magnitude uncertainty',
    'mag_obs_std': 'Estimated ACA star magnitude standard deviation',
    'color': 'Star color from the AGASC catalog',
    'n_obsids': 'Number of observations for the star',
    'n_obsids_fail': 'Number of observations which give an unexpected error',
    'n_obsids_suspect':
        'Number of observations deemed "suspect" and ignored in the magnitude estimate',
    'n_obsids_ok': 'Number of observations considered in the magnitude estimate',
    'n_no_track': 'Number of observations where the star was never tracked',
    'n': 'Total number of image samples for the star',
    'n_ok': 'Total number of image samples included in magnitude estimate for the star',
    'f_ok': 'Fraction of the total samples included in magnitude estimate',
    'median': 'Median magnitude over OK image samples',
    'sigma_minus': '15.8% quantile of magnitude over OK image samples',
    'sigma_plus': '84.2% quantile of magnitude over OK image samples',
    'mean': 'Mean of magnitude over OK image samples',
    'std': 'Standard deviation of magnitude over OK image samples',
    'mag_weighted_mean':
        'Average of magnitudes over observations, weighed by the inverse of its standard deviation',
    'mag_weighted_std':
        'Uncertainty in the weighted magnitude mean',
    't_mean': 'Mean magnitude after removing outliers on a per-observation basis',
    't_std': 'Magnitude standard deviation after removing outliers on a per-observation basis',
    'n_outlier': 'Number of outliers, removed on a per-observation basis',
    't_mean_1': 'Mean magnitude after removing 1.5*IQR outliers',
    't_std_1': 'Magnitude standard deviation after removing 1.5*IQR outliers',
    'n_outlier_1': 'Number of 1.5*IQR outliers',
    't_mean_2': 'Mean magnitude after removing 3*IQR outliers',
    't_std_2': 'Magnitude standard deviation after removing 3*IQR outliers',
    'n_outlier_2': 'Number of 3*IQR outliers',
    'selected_atol': 'abs(mag_obs - mag_aca) > 0.3',
    'selected_rtol': 'abs(mag_obs - mag_aca) > 3 * mag_aca_err',
    'selected_mag_aca_err': 'mag_aca_err > 0.2',
    'selected_color': '(color == 1.5) | (color == 0.7)',
    't_mean_dr3':
        'Truncated mean magnitude after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    't_std_dr3':
        'Truncated magnitude standard deviation after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'mean_dr3':
        'Mean magnitude after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'std_dr3':
        'Magnitude standard deviation after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'f_dr3': 'Fraction of OK image samples with angular residual less than 3 arcsec',
    'n_dr3': 'Number of OK image samples with angular residual less than 3 arcsec',
    'n_dr3_outliers':
        'Number of magnitude outliers after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'median_dr3':
        'Median magnitude after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'sigma_minus_dr3':
        '15.8% quantile of magnitude after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',
    'sigma_plus_dr3':
        '84.2% quantile of magnitude after removing outliers and samples with '
        'angular residual > 3 arcsec on a per-observation basis',

    't_mean_dr5':
        'Truncated mean magnitude after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    't_std_dr5':
        'Truncated magnitude standard deviation after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'mean_dr5':
        'Mean magnitude after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'std_dr5':
        'Magnitude standard deviation after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'f_dr5': 'Fraction of OK image samples with angular residual less than 5 arcsec',
    'n_dr5': 'Number of OK image samples with angular residual less than 5 arcsec',
    'n_dr5_outliers':
        'Number of magnitude outliers after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'median_dr5':
        'Median magnitude after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'sigma_minus_dr5':
        '15.8% quantile of magnitude after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
    'sigma_plus_dr5':
        '84.2% quantile of magnitude after removing outliers and samples with '
        'angular residual > 5 arcsec on a per-observation basis',
}


def get_agasc_id_stats(agasc_id, obs_status_override=None, tstop=None):
    """
    Get summary magnitude statistics for an AGASC ID.

    :param agasc_id: int
    :param obs_status_override: dict.
        Dictionary overriding the OK flag for specific observations.
        Keys are (OBSID, AGASC ID) pairs, values are dictionaries like
        {'obs_ok': True, 'comments': 'some comment'}
    :param tstop: cxotime-compatible timestamp
        Only entries in catalogs.STARS_OBS prior to this timestamp are considered.
    :return: dict
        dictionary with stats
    """
    logger.debug(f'Getting stats for AGASC ID {agasc_id}...')
    min_mag_obs_err = 0.03
    if not obs_status_override:
        obs_status_override = {}

    star_obs_catalogs.load(tstop=tstop)
    # Get a table of every time the star has been observed
    idx0, idx1 = star_obs_catalogs.STARS_OBS_MAP[agasc_id]
    star_obs = star_obs_catalogs.STARS_OBS[idx0:idx1]
    if len(star_obs) > 1:
        star_obs = star_obs.loc['mp_starcat_time', sorted(star_obs['mp_starcat_time'])]

    # this is the default result, if nothing gets calculated
    result = {
        'last_obs_time': 0,
        'agasc_id': agasc_id,
        'mag_aca': np.nan,
        'mag_aca_err': np.nan,
        'mag_obs': 0.,
        'mag_obs_err': np.nan,
        'mag_obs_std': 0.,
        'color': np.nan,
        'n_obsids': 0,
        'n_obsids_fail': 0,
        'n_obsids_suspect': 0,
        'n_obsids_ok': 0,
        'n_no_track': 0,
        'n': 0,
        'n_ok': 0,
        'f_ok': 0.,
        'median': 0,
        'sigma_minus': 0,
        'sigma_plus': 0,
        'mean': 0,
        'std': 0,
        'mag_weighted_mean': 0,
        'mag_weighted_std': 0,
        't_mean': 0,
        't_std': 0,
        'n_outlier': 0,
        't_mean_1': 0,
        't_std_1': 0,
        'n_outlier_1': 0,
        't_mean_2': 0,
        't_std_2': 0,
        'n_outlier_2': 0,
        # these are the criteria for including in supplement
        'selected_atol': False,
        'selected_rtol': False,
        'selected_mag_aca_err': False,
        'selected_color': False
    }

    for dr in [3, 5]:
        result.update({
            f't_mean_dr{dr}': 0,
            f't_std_dr{dr}': 0,
            f't_mean_dr{dr}_not': 0,
            f't_std_dr{dr}_not': 0,
            f'mean_dr{dr}': 0,
            f'std_dr{dr}': 0,
            f'f_dr{dr}': 0,
            f'n_dr{dr}': 0,
            f'n_dr{dr}_outliers': 0,
            f'median_dr{dr}': 0,
            f'sigma_minus_dr{dr}': 0,
            f'sigma_plus_dr{dr}': 0,
        })

    n_obsids = len(star_obs)

    # exclude star_obs that are in obs_status_override with status != 0
    excluded_obs = np.array([((oi, ai) in obs_status_override
                             and obs_status_override[(oi, ai)]['status'] != 0)
                             for oi, ai in star_obs[['mp_starcat_time', 'agasc_id']]])
    if np.any(excluded_obs):
        logger.debug('  Excluding observations flagged in obs-status table: '
                     f'{list(star_obs[excluded_obs]["obsid"])}')

    included_obs = np.array([((oi, ai) in obs_status_override
                             and obs_status_override[(oi, ai)]['status'] == 0)
                             for oi, ai in star_obs[['mp_starcat_time', 'agasc_id']]])
    if np.any(included_obs):
        logger.debug('  Including observations marked OK in obs-status table: '
                     f'{list(star_obs[included_obs]["obsid"])}')

    failures = []
    all_telem = []
    stats = []
    last_obs_time = 0
    for i, obs in enumerate(star_obs):
        oi, ai = obs['mp_starcat_time', 'agasc_id']
        comment = ''
        if (oi, ai) in obs_status_override:
            status = obs_status_override[(oi, ai)]
            logger.debug(f'  overriding status for (AGASC ID {ai}, starcat time {oi}): '
                         f'{status["status"]}, {status["comments"]}')
            comment = status['comments']
        try:
            last_obs_time = CxoTime(obs['mp_starcat_time']).cxcsec
            telem = Table(get_telemetry(obs))
            obs_stat = get_obs_stats(obs, telem={k: telem[k] for k in telem.colnames})
            obs_stat.update({
                'obs_ok': (
                    included_obs[i] | (
                        ~excluded_obs[i]
                        & (obs_stat['n'] > 10)
                        & (obs_stat['f_track'] > 0.3)
                        & (obs_stat['lf_variability_100s'] < 1)
                    )
                ),
                'obs_suspect': False,
                'obs_fail': False,
                'comments': comment
            })
            all_telem.append(telem)
            stats.append(obs_stat)

            if not obs_stat['obs_ok'] and not excluded_obs[i]:
                obs_stat['obs_suspect'] = True
                failures.append(
                    dict(MagStatsException(msg='Suspect observation',
                                           agasc_id=obs['agasc_id'],
                                           obsid=obs['obsid'],
                                           mp_starcat_time=obs["mp_starcat_time"],)))
        except MagStatsException as e:
            # this except branch deals with exceptions thrown by get_telemetry
            all_telem.append(None)
            # length-zero telemetry short-circuits any new call to get_telemetry
            obs_stat = get_obs_stats(obs, telem=[])
            obs_stat.update({
                'obs_ok': False,
                'obs_suspect': False,
                'obs_fail': True,
                'comments': comment if excluded_obs[i] else f'Error: {e.msg}.'
            })
            stats.append(obs_stat)
            if not excluded_obs[i]:
                logger.debug(
                    f'  Error in get_agasc_id_stats({agasc_id=}, obsid={obs["obsid"]}): {e}')
                failures.append(dict(e))

    stats = Table(stats)
    stats['w'] = np.nan
    stats['mean_corrected'] = np.nan
    stats['weighted_mean'] = np.nan

    star = get_star(agasc_id, use_supplement=False)

    result.update({
        'last_obs_time': last_obs_time,
        'mag_aca': star['MAG_ACA'],
        'mag_aca_err': star['MAG_ACA_ERR'] / 100,
        'color': star['COLOR1'],
        'n_obsids_fail': len(failures),
        'n_obsids_suspect': np.sum(stats['obs_suspect']),
        'n_obsids': n_obsids,
    })

    if not np.any(~excluded_obs):
        # this happens when all observations have been flagged as not OK a priory (obs-status).
        logger.debug(f'  Skipping star in get_agasc_id_stats({agasc_id=}).'
                     ' All observations are flagged as not good.')
        return result, stats, failures

    if len(all_telem) - len(failures) <= 0:
        # and we reach here if some observations were not flagged as bad, but all failed.
        logger.debug(f'  Error in get_agasc_id_stats({agasc_id=}):'
                     ' There is no OK observation.')
        return result, stats, failures

    excluded_obs += np.array([t is None for t in all_telem])

    logger.debug('  identifying outlying observations...')
    for i, (s, t) in enumerate(zip(stats, all_telem)):
        if excluded_obs[i]:
            continue
        t['obs_ok'] = np.ones_like(t['ok'], dtype=bool) * s['obs_ok']
        logger.debug('  identifying outlying observations '
                     f'(OBSID={s["obsid"]}, mp_starcat_time={s["mp_starcat_time"]})')
        t['obs_outlier'] = np.zeros_like(t['ok'])
        if np.any(t['ok']) and s['f_track'] > 0 and s['obs_ok']:
            iqr = s['q75'] - s['q25']
            t['obs_outlier'] = (
                t['ok']
                & (iqr > 0)
                & ((t['mags'] < s['q25'] - 1.5 * iqr) | (t['mags'] > s['q75'] + 1.5 * iqr))
            )
    all_telem = vstack([Table(t) for i, t in enumerate(all_telem) if not excluded_obs[i]])
    kalman = (all_telem['AOACASEQ'] == 'KALM') & (all_telem['AOPCADMD'] == 'NPNT')
    all_telem = all_telem[kalman]  # non-npm/non-kalman are excluded

    mags = all_telem['mags']
    ok = all_telem['ok'] & all_telem['obs_ok']

    f_ok = np.sum(ok) / len(ok)

    result.update({
        'mag_obs_err': min_mag_obs_err,
        'n_obsids_ok': np.sum(stats['obs_ok']),
        'n_no_track': (
            np.sum((~stats['obs_ok']))
            + np.sum(stats['f_ok'][stats['obs_ok']] < 0.3)
        ),
        'n': len(ok),
        'n_ok': np.sum(ok),
        'f_ok': f_ok,
    })

    if result['n_ok'] < 10:
        return result, stats, failures

    sigma_minus, q25, median, q75, sigma_plus = np.quantile(mags[ok],
                                                            [0.158, 0.25, 0.5, 0.75, 0.842])
    iqr = q75 - q25
    outlier_1 = ok & ((mags > q75 + 1.5 * iqr) | (mags < q25 - 1.5 * iqr))
    outlier_2 = ok & ((mags > q75 + 3 * iqr) | (mags < q25 - 3 * iqr))
    outlier = all_telem['obs_outlier']

    # combine measurements using a weighted mean
    obs_ok = stats['obs_ok']
    min_std = max(0.1, stats[obs_ok]['std'].min())
    stats['w'][obs_ok] = np.where(stats['std'][obs_ok] != 0,
                                  1. / stats['std'][obs_ok],
                                  1. / min_std)
    stats['mean_corrected'][obs_ok] = stats['t_mean'][obs_ok] + stats['mag_correction'][obs_ok]
    stats['weighted_mean'][obs_ok] = stats['mean_corrected'][obs_ok] * stats['w'][obs_ok]

    mag_weighted_mean = (stats[obs_ok]['weighted_mean'].sum() / stats[obs_ok]['w'].sum())
    mag_weighted_std = (
        np.sqrt(((stats[obs_ok]['mean'] - mag_weighted_mean)**2 * stats[obs_ok]['w']).sum()
                / stats[obs_ok]['w'].sum())
    )

    result.update({
        'agasc_id': agasc_id,
        'n': len(ok),
        'n_ok': np.sum(ok),
        'f_ok': f_ok,
        'median': median,
        'sigma_minus': sigma_minus,
        'sigma_plus': sigma_plus,
        'mean': np.mean(mags[ok]),
        'std': np.std(mags[ok]),
        'mag_weighted_mean': mag_weighted_mean,
        'mag_weighted_std': mag_weighted_std,
        't_mean': np.mean(mags[ok & (~outlier)]),
        't_std': np.std(mags[ok & (~outlier)]),
        'n_outlier': np.sum(ok & outlier),
        't_mean_1': np.mean(mags[ok & (~outlier_1)]),
        't_std_1': np.std(mags[ok & (~outlier_1)]),
        'n_outlier_1': np.sum(ok & outlier_1),
        't_mean_2': np.mean(mags[ok & (~outlier_2)]),
        't_std_2': np.std(mags[ok & (~outlier_2)]),
        'n_outlier_2': np.sum(ok & outlier_2),
    })

    for dr in [3, 5]:
        k = ok & (all_telem['dy'] < dr & all_telem['dz'] < dr)
        k2 = ok & (all_telem['dy'] >= dr | all_telem['dz'] >= dr)
        if not np.any(k):
            continue
        sigma_minus, q25, median, q75, sigma_plus = np.quantile(mags[k],
                                                                [0.158, 0.25, 0.5, 0.75, 0.842])
        outlier = ok & all_telem['obs_outlier']
        mag_not = np.nanmean(mags[k2 & (~outlier)]) if np.sum(k2 & (~outlier)) else np.nan
        std_not = np.nanstd(mags[k2 & (~outlier)]) if np.sum(k2 & (~outlier)) else np.nan
        result.update({
            f't_mean_dr{dr}': np.mean(mags[k & (~outlier)]),
            f't_std_dr{dr}': np.std(mags[k & (~outlier)]),
            f't_mean_dr{dr}_not': mag_not,
            f't_std_dr{dr}_not': std_not,
            f'mean_dr{dr}': np.mean(mags[k]),
            f'std_dr{dr}': np.std(mags[k]),
            f'f_dr{dr}': np.sum(k) / np.sum(ok),
            f'n_dr{dr}': np.sum(k),
            f'n_dr{dr}_outliers': np.sum(k & outlier),
            f'f_ok_{dr}': np.sum(k) / len(k),
            f'median_dr{dr}': median,
            f'sigma_minus_dr{dr}': sigma_minus,
            f'sigma_plus_dr{dr}': sigma_plus,
        })

    result.update({
        'mag_obs': result['t_mean_dr5'],
        'mag_obs_err': np.sqrt(result['t_std_dr5']**2 + min_mag_obs_err**2),
        'mag_obs_std': result['t_std_dr5'],
    })

    # these are the criteria for including in supplement
    result.update({
        'selected_atol': np.abs(result['mag_obs'] - result['mag_aca']) > 0.3,
        'selected_rtol': np.abs(result['mag_obs'] - result['mag_aca']) > 3 * result['mag_aca_err'],
        'selected_mag_aca_err': result['mag_aca_err'] > 0.2,
        'selected_color': (result['color'] == 1.5) | (np.isclose(result['color'], 0.7))
    })

    logger.debug(f'  stats for AGASC ID {agasc_id}: '
                 f' {stats["mag_obs"][0]}')
    return result, stats, failures
