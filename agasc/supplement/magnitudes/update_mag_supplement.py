#!/usr/bin/env python
import warnings
import os
import pickle
import datetime
import logging
from functools import partial
from multiprocessing import Pool

import tables
import numpy as np
from astropy import table
from astropy import time, units as u

from agasc.supplement.magnitudes import star_obs_catalogs, mag_estimate, mag_estimate_report as msr
from cxotime import CxoTime


logger = logging.getLogger('agasc.supplement')


def level0_archive_time_range():
    """
    Return the time range covered by mica archive aca_l0 files.

    :return: tuple of CxoTime
    """
    import sqlite3
    import os
    db_file = os.path.expandvars('$SKA/data/mica/archive/aca0/archfiles.db3')
    with sqlite3.connect(db_file) as connection:
        cursor = connection.cursor()
        cursor.execute("select tstop from archfiles order by tstop desc limit 1")
        t_stop = cursor.fetchall()[0][0]
        cursor.execute("select tstop from archfiles order by tstart asc limit 1")
        t_start = cursor.fetchall()[0][0]
        return CxoTime(t_stop).date, CxoTime(t_start).date


def get_agasc_id_stats(agasc_ids, obs_status_override={}, tstop=None):
    """
    Call mag_stats.get_agasc_id_stats for each AGASC ID

    :param agasc_ids: list
    :param obs_status_override: dict.
        Dictionary overriding the OK flag for specific observations.
        Keys are (OBSID, AGASC ID) pairs, values are dictionaries like
        {'obs_ok': True, 'comments': 'some comment'}
    :param tstop: cxotime-compatible timestamp
        Only observations prior to this timestamp are considered.
    :return: astropy.table.Table, astropy.table.Table, list
        obs_stats, agasc_stats, fails
    """
    from agasc.supplement.magnitudes import mag_estimate
    from astropy.table import Table, vstack

    fails = []
    obs_stats = []
    agasc_stats = []
    for i, agasc_id in enumerate(agasc_ids):
        try:
            agasc_stat, obs_stat, obs_fail = \
                mag_estimate.get_agasc_id_stats(agasc_id=agasc_id,
                                                obs_status_override=obs_status_override,
                                                tstop=tstop)
            agasc_stats.append(agasc_stat)
            obs_stats.append(obs_stat)
            fails += obs_fail
        except mag_estimate.MagStatsException as e:
            fails.append(dict(e))
        except Exception as e:
            # transform Exception to MagStatsException for standard book keeping
            fails.append(dict(mag_estimate.MagStatsException(agasc_id=agasc_id, msg=str(e))))

    try:
        agasc_stats = Table(agasc_stats) if agasc_stats else None
        obs_stats = vstack(obs_stats) if obs_stats else None
    except Exception as e:
        agasc_stats = None
        obs_stats = None
        # transform Exception to MagStatsException for standard book keeping
        fails.append(dict(mag_estimate.MagStatsException(
            msg=f'Exception at end of get_agasc_id_stats: {str(e)}')))

    return obs_stats, agasc_stats, fails


def get_agasc_id_stats_pool(agasc_ids, obs_status_override=None, batch_size=100, tstop=None):
    """
    Call update_mag_stats.get_agasc_id_stats multiple times using a multiprocessing.Pool

    :param agasc_ids: list
    :param obs_status_override: dict.
        Dictionary overriding the OK flag for specific observations.
        Keys are (OBSID, AGASC ID) pairs, values are dictionaries like
        {'obs_ok': True, 'comments': 'some comment'}
    :param batch_size: int
    :param tstop: cxotime-compatible timestamp
        Only observations prior to this timestamp are considered.
    :return: astropy.table.Table, astropy.table.Table, list
        obs_stats, agasc_stats, fails, failed_jobs
    """
    import time
    from astropy.table import vstack, Table

    if obs_status_override is None:
        obs_status_override = {}

    fmt = '%Y-%m-%d %H:%M'
    jobs = []
    n = len(agasc_ids)
    args = []
    progress = 0
    finished = 0
    for i in range(0, n, batch_size):
        args.append(agasc_ids[i:i + batch_size])
    with Pool() as pool:
        for arg in args:
            jobs.append(pool.apply_async(get_agasc_id_stats,
                                         [arg, obs_status_override, tstop]))
        start = datetime.datetime.now()
        now = None
        while finished < len(jobs):
            finished = sum([f.ready() for f in jobs])
            if now is None or 100 * finished / len(jobs) - progress > 0.02:
                now = datetime.datetime.now()
                if finished == 0:
                    eta = ''
                else:
                    dt1 = (now - start).total_seconds()
                    dt = datetime.timedelta(seconds=(len(jobs) - finished) * dt1 / finished)
                    eta = f'ETA: {(now + dt).strftime(fmt)}'
                progress = 100 * finished / len(jobs)
                logger.info(f'{progress:6.2f}% at {now.strftime(fmt)}, {eta}')
            time.sleep(1)
    fails = []
    failed_agasc_ids = [i for arg, job in zip(args, jobs) if not job.successful() for i in arg]
    for agasc_id in failed_agasc_ids:
        fails.append(dict(mag_estimate.MagStatsException(agasc_id=agasc_id, msg='Failed job')))

    results = [job.get() for job in jobs if job.successful()]

    obs_stats = [r[0] for r in results if r[0] is not None]
    agasc_stats = [r[1] for r in results if r[1] is not None]
    obs_stats = vstack(obs_stats) if obs_stats else Table()
    agasc_stats = vstack(agasc_stats) if agasc_stats else Table()
    fails += sum([r[2] for r in results], [])

    return obs_stats, agasc_stats, fails


def _update_table(table_old, table_new, keys):
    # checking names, because actual types change upon saving in fits format
    assert table_old.as_array().dtype.names == table_new.as_array().dtype.names, \
        'Tables have different dtype'
    table_old = table_old.copy()
    new_row = np.ones(len(table_new), dtype=bool)
    _, i_new, i_old = np.intersect1d(table_new[keys].as_array(),
                                     table_old[keys].as_array(),
                                     return_indices=True)
    new_row[i_new] = False
    table_old[i_old] = table_new[i_new]
    return table.vstack([table_old, table_new[new_row]])


def update_mag_stats(obs_stats, agasc_stats, fails, outdir='.'):
    """
    Update the mag_stats catalog.

    I currently save three files:
    - mag_stats_agasc.fits with stats for each AGASC ID
    - mag_stats_obsid.fits with stats for each OBSID
    - mag_stats_fails.pkl with a list of failures

    :param obs_stats:
    :param agasc_stats:
    :param fails:
    :param outdir:
    :return:
    """
    if agasc_stats is not None and len(agasc_stats):
        filename = outdir / 'mag_stats_agasc.fits'
        if filename.exists():
            agasc_stats = _update_table(table.Table.read(filename), agasc_stats,
                                        keys=['agasc_id'])
            os.remove(filename)
        agasc_stats.write(filename)
    if obs_stats is not None and len(obs_stats):
        filename = outdir / 'mag_stats_obsid.fits'
        if filename.exists():
            obs_stats = _update_table(table.Table.read(filename), obs_stats,
                                      keys=['agasc_id', 'obsid', 'timeline_id'])
            os.remove(filename)
        obs_stats.write(filename)
    if len(fails):
        filename = outdir / 'mag_stats_fails.pkl'
        with open(filename, 'wb') as out:
            pickle.dump(fails, out)


def update_supplement(agasc_stats, filename, include_all=True):
    """
    Update the magnitude table of the AGASC supplement.

    :param agasc_stats:
    :param filename:
    :param include_all: bool
        if True, all OK entries are included in supplement.
        if False, only OK entries marked 'selected_*'
    :return:
    """
    mags_dtype = np.dtype([('agasc_id', np.int32),
                           ('mag_aca', np.float32),
                           ('mag_aca_err', np.float32),
                           ('last_obs_time', np.float64)])

    if include_all:
        outliers_new = agasc_stats[
            (agasc_stats['n_obsids_ok'] > 0)
        ]
    else:
        outliers_new = agasc_stats[
            (agasc_stats['n_obsids_ok'] > 0)
            & (agasc_stats['selected_atol']
               | agasc_stats['selected_rtol']
               | agasc_stats['selected_color']
               | agasc_stats['selected_mag_aca_err'])
        ]
    outliers_new['mag_aca'] = outliers_new['mag_obs']
    outliers_new['mag_aca_err'] = outliers_new['mag_obs_err']

    outliers_new = outliers_new[mags_dtype.names].as_array()
    if outliers_new.dtype != mags_dtype:
        outliers_new = outliers_new.astype(mags_dtype)

    outliers = None
    new_stars = None
    updated_stars = None
    if filename.exists():
        # I could do what follows directly in place, but the table is not that large.
        with tables.File(filename, 'r') as h5:
            if 'mags' in h5.root:
                outliers_current = h5.root.mags[:]
                # find the indices of agasc_ids in both current and new lists
                _, i_new, i_cur = np.intersect1d(outliers_new['agasc_id'],
                                                 outliers_current['agasc_id'],
                                                 return_indices=True)
                current = outliers_current[i_cur]
                new = outliers_new[i_new]

                # from those, find the ones which differ in last observation time
                i_cur = i_cur[current['last_obs_time'] != new['last_obs_time']]
                i_new = i_new[current['last_obs_time'] != new['last_obs_time']]
                # overwrite current values with new values (and calculate diff to return)
                updated_stars = np.zeros(len(outliers_new[i_new]),
                                         dtype=mags_dtype)
                updated_stars['mag_aca'] = (outliers_new[i_new]['mag_aca']
                                            - outliers_current[i_cur]['mag_aca'])
                updated_stars['mag_aca_err'] = (outliers_new[i_new]['mag_aca_err']
                                                - outliers_current[i_cur]['mag_aca_err'])
                updated_stars['agasc_id'] = outliers_new[i_new]['agasc_id']
                outliers_current[i_cur] = outliers_new[i_new]

                # find agasc_ids in new list but not in current list
                new_stars = ~np.in1d(outliers_new['agasc_id'], outliers_current['agasc_id'])
                # and add them to the current list
                outliers_current = np.concatenate([outliers_current, outliers_new[new_stars]])
                outliers = np.sort(outliers_current)

                new_stars = outliers_new[new_stars]['agasc_id']

    if outliers is None:
        outliers = outliers_new
        new_stars = outliers_new['agasc_id']
        updated_stars = np.array([], dtype=mags_dtype)

    mode = 'r+' if filename.exists() else 'w'
    with tables.File(filename, mode) as h5:
        if 'mags' in h5.root:
            h5.remove_node('/mags')
        h5.create_table('/', 'mags', outliers)

    return new_stars, updated_stars


def do(output_dir,
       reports_dir,
       agasc_ids=None,
       multi_process=False,
       start=None,
       stop=None,
       whole_history=False,
       report=False,
       email='',
       include_bad=False,
       dry_run=False):
    """

    :param output_dir:
    :param reports_dir:
    :param agasc_ids:
    :param multi_process:
    :param start:
    :param stop:
    :param whole_history:
    :param report:
    :param email:
    :return:
    """
    # PyTables is not really unicode-compatible, but python 3 is basically unicode.
    # For our purposes, PyTables works. It would fail with characters that can not be written
    # as ascii. It displays a warning which I want to avoid:
    warnings.filterwarnings("ignore", category=tables.exceptions.FlavorWarning)

    filename = output_dir / 'agasc_supplement.h5'

    if multi_process:
        get_stats = partial(get_agasc_id_stats_pool, batch_size=10)
    else:
        get_stats = get_agasc_id_stats

    if agasc_ids is not None:
        agasc_ids = np.intersect1d(agasc_ids, star_obs_catalogs.STARS_OBS['agasc_id'])

    # set start/stop times and agasc_ids
    if whole_history or agasc_ids is not None:
        if start or stop:
            raise ValueError('incompatible arguments: whole_history and start/stop')
        start = CxoTime(star_obs_catalogs.STARS_OBS['mp_starcat_time']).min().date
        stop = CxoTime(star_obs_catalogs.STARS_OBS['mp_starcat_time']).max().date
        if agasc_ids is None:
            agasc_ids = sorted(star_obs_catalogs.STARS_OBS['agasc_id'])
    else:
        stop = CxoTime(stop).date if stop else CxoTime.now().date
        start = CxoTime(start).date if start else (CxoTime(stop) - 14 * u.day).date
        obs_in_time = ((star_obs_catalogs.STARS_OBS['mp_starcat_time'] >= start)
                       & (star_obs_catalogs.STARS_OBS['mp_starcat_time'] <= stop))
        agasc_ids = sorted(star_obs_catalogs.STARS_OBS[obs_in_time]['agasc_id'])

    agasc_ids = np.unique(agasc_ids)
    stars_obs = star_obs_catalogs.STARS_OBS[
        np.in1d(star_obs_catalogs.STARS_OBS['agasc_id'], agasc_ids)
    ]

    # if supplement exists:
    # - drop bad stars
    # - get OBS status override
    # - get the latest observation for each agasc_id,
    # - find the ones already in the supplement
    # - include only the ones with supplement.last_obs_time < than stars_obs.mp_starcat_time
    obs_status = {}
    if filename.exists():
        with tables.File(filename, 'r') as h5:
            if not include_bad and 'bad' in h5.root:
                logger.info('Excluding bad stars')
                stars_obs = stars_obs[~np.in1d(stars_obs['agasc_id'], h5.root.bad[:]['agasc_id'])]

            if 'obs' in h5.root:
                obs_status = {
                    (r['obsid'], r['agasc_id']): {'ok': r['ok'], 'comments': r['comments']}
                    for r in h5.root.obs[:]
                }
            if 'mags' in h5.root:
                outliers_current = h5.root.mags[:]
                times = stars_obs[['agasc_id', 'mp_starcat_time']].group_by(
                    'agasc_id').groups.aggregate(lambda d: np.max(CxoTime(d)).date)
                if len(outliers_current):
                    times = table.join(times,
                                       table.Table(outliers_current[['agasc_id', 'last_obs_time']]),
                                       join_type='left')
                else:
                    times['last_obs_time'] = table.MaskedColumn(
                        np.zeros(len(times), dtype=h5.root.mags.dtype['last_obs_time']),
                        mask=np.ones(len(times), dtype=bool)
                    )
                if hasattr(times['last_obs_time'], 'mask'):
                    # the mask exists if there are stars in stars_obs
                    # that are not in outliers_current
                    update = (times['last_obs_time'].mask
                              | ((~times['last_obs_time'].mask)
                                 & (CxoTime(times['mp_starcat_time']).cxcsec
                                    > times['last_obs_time']).data)
                              )
                else:
                    update = (CxoTime(times['mp_starcat_time']).cxcsec > times['last_obs_time'])

                stars_obs = stars_obs[np.in1d(stars_obs['agasc_id'], times[update]['agasc_id'])]
                agasc_ids = np.sort(np.unique(stars_obs['agasc_id']))
                if len(update) - np.sum(update):
                    logger.info(f'Skipping {len(update) - np.sum(update)} '
                                f'stars already in the supplement')

    # do the processing
    logger.info(f'Will process {len(agasc_ids)} stars on {len(stars_obs)} observations')
    logger.info(f'from {start} to {stop}')
    if dry_run:
        return

    obs_stats, agasc_stats, fails = \
        get_stats(agasc_ids, tstop=stop, obs_status_override=obs_status)

    failed_global = [f for f in fails if not f['agasc_id'] and not f['obsid']]
    failed_stars = [f for f in fails if f['agasc_id'] and not f['obsid']]
    failed_obs = [f for f in fails if f['obsid']]
    logger.info(
        f'Got:\n'
        f'  {0 if obs_stats is None else len(obs_stats)} OBSIDs,'
        f'  {0 if agasc_stats is None else len(agasc_stats)} stars,'
        f'  {len(failed_stars)} failed stars,'
        f'  {len(failed_obs)} failed observations,'
        f'  {len(failed_global)} global errors'
    )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    update_mag_stats(obs_stats, agasc_stats, fails, output_dir)
    if agasc_stats is not None and len(agasc_stats):
        new_stars, updated_stars = update_supplement(agasc_stats,
                                                     filename=filename)

        logger.info(f'  {len(new_stars)} new stars, {len(updated_stars)} updated stars')
        if email:
            try:
                bad_obs = (
                    (obs_stats['mp_starcat_time'] >= start)
                    & (obs_stats['mp_starcat_time'] < stop)
                    & ~obs_stats['obs_ok']
                )
                if np.any(bad_obs):
                    msr.email_bad_obs_report(obs_stats[bad_obs], to=email)
            except Exception as e:
                logger.error(f'Failed sending email to {email}: {e}')

        if report:
            now = datetime.datetime.now()
            logger.info(f"making report at {now}")
            sections = [{
                'id': 'new_stars',
                'title': 'New Stars',
                'stars': new_stars
            }, {
                'id': 'updated_stars',
                'title': 'Updated Stars',
                'stars': updated_stars['agasc_id'] if len(updated_stars) else []
            }]

            week = time.TimeDelta(7 * u.day)
            t = CxoTime(stop)
            nav_links = {
                'previous': f'../{(t - week).date[:8]}/index.html',
                'up': '..',
                'next': f'../{(t + week).date[:8]}/index.html'
            }

            directory = reports_dir / f'{t.date[:8]}'
            multi_star_html_args = dict(
                filename='index.html',
                sections=sections,
                updated_stars=updated_stars,
                fails=fails,
                report_date=CxoTime.now().date,
                tstart=start,
                tstop=stop,
                nav_links=nav_links,
                include_all_stars=True
            )
            try:
                report = msr.MagEstimateReport(agasc_stats, obs_stats, directory=directory)
                report.multi_star_html(**multi_star_html_args)
                latest = reports_dir / 'latest'
                if latest.exists():
                    latest.unlink()
                latest.symlink_to(directory)
            except Exception as e:
                logger.error(f'Exception when creating report: {e}')
                multi_star_html_args['directory'] = directory
                failure_file = output_dir / f'failed_report_{t.date[:8]}.pkl'
                with open(failure_file, 'wb') as fh:
                    pickle.dump(multi_star_html_args, fh)
                logger.error(f'Intermediate data saved in {failure_file}')

    now = datetime.datetime.now()
    logger.info(f"done at {now}")
