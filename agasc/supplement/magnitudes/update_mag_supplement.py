#!/usr/bin/env python
import warnings
import os
import pickle
import datetime
import logging
from functools import partial
from multiprocessing import Pool
import jinja2

from tqdm import tqdm
import tables
import numpy as np
from astropy import table
from astropy import time, units as u

from agasc.supplement.magnitudes import star_obs_catalogs, mag_estimate, mag_estimate_report as msr
from agasc.supplement.utils import save_version, MAGS_DTYPE
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


def get_agasc_id_stats(agasc_ids, obs_status_override={}, tstop=None, no_progress=None):
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
    bar = tqdm(agasc_ids, desc='progress', disable=no_progress, unit='star')
    for agasc_id in agasc_ids:
        bar.update()
        try:
            logger.debug('-' * 80)
            logger.debug(f'{agasc_id=}')
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
            logger.debug(f'Unexpected Error: {e}')
            fails.append(dict(mag_estimate.MagStatsException(agasc_id=agasc_id, msg=str(e))))
            raise
    bar.close()
    logger.debug('-' * 80)

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


def get_agasc_id_stats_pool(agasc_ids, obs_status_override=None, batch_size=100, tstop=None,
                            no_progress=None):
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

    jobs = []
    args = []
    finished = 0
    logger.info(f'Processing {batch_size} stars per job')
    for i in range(0, len(agasc_ids), batch_size):
        args.append(agasc_ids[i:i + batch_size])
    with Pool() as pool:
        for arg in args:
            jobs.append(pool.apply_async(get_agasc_id_stats,
                                         [arg, obs_status_override, tstop, True]))
        bar = tqdm(total=len(jobs), desc='progress', disable=no_progress, unit='job')
        while finished < len(jobs):
            finished = sum([f.ready() for f in jobs])
            if finished - bar.n:
                bar.update(finished - bar.n)
            time.sleep(1)
        bar.close()

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
        logger.debug(f'Updating {filename}')
        if filename.exists():
            agasc_stats = _update_table(table.Table.read(filename), agasc_stats,
                                        keys=['agasc_id'])
            os.remove(filename)
        agasc_stats.write(filename)
    if obs_stats is not None and len(obs_stats):
        filename = outdir / 'mag_stats_obsid.fits'
        logger.debug(f'Updating {filename}')
        if filename.exists():
            obs_stats = _update_table(table.Table.read(filename), obs_stats,
                                      keys=['agasc_id', 'obsid', 'timeline_id'])
            os.remove(filename)
        obs_stats.write(filename)
    if len(fails):
        filename = outdir / 'mag_stats_fails.pkl'
        logger.debug(f'Updating {filename}')
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

    outliers_new = outliers_new[MAGS_DTYPE.names].as_array()
    if outliers_new.dtype != MAGS_DTYPE:
        outliers_new = outliers_new.astype(MAGS_DTYPE)

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
                                         dtype=MAGS_DTYPE)
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
        logger.warning('Creating new "mags" table')
        outliers = outliers_new
        new_stars = outliers_new['agasc_id']
        updated_stars = np.array([], dtype=MAGS_DTYPE)

    mode = 'r+' if filename.exists() else 'w'
    with tables.File(filename, mode) as h5:
        if 'mags' in h5.root:
            h5.remove_node('/mags')
        h5.create_table('/', 'mags', outliers)
    save_version(filename, 'mags')

    return new_stars, updated_stars


def write_obs_status_yaml(obs_stats=None, fails=(), filename=None):
    obs = []
    if obs_stats and len(obs_stats):
        obs_stats = obs_stats[~obs_stats['obs_ok']]
        observation_ids = np.unique(obs_stats['observation_id'])
        for observation_id in observation_ids:
            rows = obs_stats[obs_stats['observation_id'] == observation_id]
            rows.sort(keys='agasc_id')
            obs.append({
                'observation_id': observation_id,
                'agasc_id': list(rows['agasc_id']),
                'status': 1,
                'comments': obs_stats['comment']
            })
    for fail in fails:
        if fail['agasc_id'] is None or fail['observation_id'] is None:
            continue
        observation_ids = fail['observation_id'] if type(fail['observation_id']) is list \
            else [fail['observation_id']]
        agasc_id = fail['agasc_id']
        for observation_id in observation_ids:
            obs.append({
                'observation_id': observation_id,
                'agasc_id': [agasc_id],
                'status': 1,
                'comments': fail['msg']
            })
    if len(obs) == 0:
        return

    yaml_template = """obs:
  {%- for obs in observations %}
  - observation_id: {{ obs.observation_id }}
    status: {{ obs.status }}
    agasc_id: [{% for agasc_id in obs.agasc_id -%}
                  {{ agasc_id }}{%- if not loop.last %}, {% endif -%}
               {%- endfor -%}]
    comments: {{ obs.comments }}
  {%- endfor %}
  """
    tpl = jinja2.Template(yaml_template)
    if filename:
        with open(filename, 'w') as fh:
            fh.write(tpl.render(observations=obs))
    return tpl.render(observations=obs)


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
       dry_run=False,
       no_progress=None):
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

    skip = True
    if agasc_ids is not None:
        agasc_ids = np.intersect1d(agasc_ids, star_obs_catalogs.STARS_OBS['agasc_id'])
        skip = False

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
    obs_status_override = {}
    if filename.exists():
        with tables.File(filename, 'r') as h5:
            if not include_bad and 'bad' in h5.root:
                logger.info('Excluding bad stars')
                stars_obs = stars_obs[~np.in1d(stars_obs['agasc_id'], h5.root.bad[:]['agasc_id'])]

            if 'obs' in h5.root:
                obs_status_override = table.Table(h5.root.obs[:])
                obs_status_override.convert_bytestring_to_unicode()
                obs_status_override = {
                    (r['observation_id'], r['agasc_id']):
                        {'status': r['status'], 'comments': r['comments']}
                    for r in obs_status_override
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
                if skip:
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
        get_stats(agasc_ids, tstop=stop,
                  obs_status_override=obs_status_override,
                  no_progress=no_progress)

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

    obs_status_file = output_dir / 'obs_status.yml'
    try:
        write_obs_status_yaml([], fails=failed_obs + failed_stars, filename=obs_status_file)
    except Exception as e:
        logger.warning(f'Failed to write {obs_status_file}: {e}')

    new_stars, updated_stars = update_supplement(agasc_stats, filename=filename)
    logger.info(f'  {len(new_stars)} new stars, {len(updated_stars)} updated stars')

    if agasc_stats is not None and len(agasc_stats):
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
                logger.error(f'Error sending email to {email}: {e}')

    if report:
        report_date = CxoTime(stop)
        # the nominal date for reports is the first Monday after the stop date.
        report_date += ((7 - report_date.datetime.weekday()) % 7) * u.day

        directory = reports_dir / f'{report_date.date[:8]}'
        report_data_dir = directory
        report_data_file = report_data_dir / f'report_data_{report_date.date[:8]}.pkl'

        week = time.TimeDelta(7 * u.day)
        nav_links = {
            'previous': f'../{(report_date - week).date[:8]}/index.html',
            'up': '..',
            'next': f'../{(report_date + week).date[:8]}/index.html'
        }

        # If the report data file exists, the arguments for the report from the file are
        # modified according to the current run. Otherwise, they are created from scratch.
        if report_data_file.exists():
            with open(report_data_file, 'rb') as fh:
                report_data = pickle.load(fh)
            logger.info(f'Loading existing report data from {report_data_file}')
            multi_star_html_args = report_data['args']

            # arguments for the report are modified here
            # merge fails:
            # - from previous run, take fails that were not run just now
            # - add current fails
            multi_star_html_args['fails'] = fails
            multi_star_html_args['no_progress'] = no_progress

        else:
            sections = [{
                'id': 'new_stars',
                'title': 'New Stars',
                'stars': new_stars
            }, {
                'id': 'updated_stars',
                'title': 'Updated Stars',
                'stars': updated_stars['agasc_id'] if len(updated_stars) else []
            }, {
                'id': 'other_stars',
                'title': 'Other Stars',
                'stars': list(agasc_stats['agasc_id'][
                    ~np.in1d(agasc_stats['agasc_id'], new_stars)
                    & ~np.in1d(agasc_stats['agasc_id'], updated_stars['agasc_id'])
                ])
            }
            ]

            multi_star_html_args = dict(
                filename='index.html',
                sections=sections,
                updated_stars=updated_stars,
                fails=fails,
                report_date=report_date.date,
                tstart=start,
                tstop=stop,
                nav_links=nav_links,
                include_all_stars=False,
                no_progress=no_progress
            )

        try:
            report = msr.MagEstimateReport(
                agasc_stats=output_dir / 'mag_stats_agasc.fits',
                obs_stats=output_dir / 'mag_stats_obsid.fits',
                directory=directory
            )
            report.multi_star_html(**multi_star_html_args)
            latest = reports_dir / 'latest'
            if os.path.lexists(latest):
                logger.debug('Removing existing "latest" symlink')
                latest.unlink()
            logger.debug('Creating "latest" symlink')
            latest.symlink_to(directory.absolute())
        except Exception as e:
            report_data_dir = output_dir
            logger.error(f'Error when creating report: {e}')
        finally:
            report_data_file = report_data_dir / report_data_file.name
            if not report_data_dir.exists():
                report_data_dir.mkdir(parents=True)
            report_data = {
                'args': multi_star_html_args,
                'directory': directory
            }
            with open(report_data_file, 'wb') as fh:
                pickle.dump(report_data, fh)
            logger.info(f'Report data saved in {report_data_file}')

    now = datetime.datetime.now()
    logger.info(f"done at {now}")
