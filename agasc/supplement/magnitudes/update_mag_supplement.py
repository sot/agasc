#!/usr/bin/env python
import datetime
import logging
import os
import pickle
import sys
import traceback
import warnings
from functools import partial
from multiprocessing import Pool

import jinja2
import numpy as np
import tables
from astropy import table, time
from astropy import units as u
from cxotime import CxoTime
from mica.starcheck import get_starcheck_catalog
from tqdm import tqdm

from agasc.supplement.magnitudes import (
    mag_estimate,
    star_obs_catalogs,
)
from agasc.supplement.magnitudes import (
    mag_estimate_report as msr,
)
from agasc.supplement.utils import MAGS_DTYPE, save_version

logger = logging.getLogger("agasc.supplement")


def level0_archive_time_range():
    """
    Return the time range covered by mica archive aca_l0 files.

    :return: tuple of CxoTime
    """
    import os
    import sqlite3

    db_file = os.path.expandvars("$SKA/data/mica/archive/aca0/archfiles.db3")
    with sqlite3.connect(db_file) as connection:
        cursor = connection.cursor()
        cursor.execute("select tstop from archfiles order by tstop desc limit 1")
        t_stop = cursor.fetchall()[0][0]
        cursor.execute("select tstop from archfiles order by tstart asc limit 1")
        t_start = cursor.fetchall()[0][0]
        return CxoTime(t_stop).date, CxoTime(t_start).date


def get_agasc_id_stats(
    agasc_ids, obs_status_override=None, tstop=None, no_progress=None
):
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
    from astropy.table import Table, vstack

    from agasc.supplement.magnitudes import mag_estimate

    if obs_status_override is None:
        obs_status_override = {}
    fails = []
    obs_stats = []
    agasc_stats = []
    bar = tqdm(agasc_ids, desc="progress", disable=no_progress, unit="star")
    for agasc_id in agasc_ids:
        bar.update()
        try:
            logger.debug("-" * 80)
            logger.debug(f"{agasc_id=}")
            agasc_stat, obs_stat, obs_fail = mag_estimate.get_agasc_id_stats(
                agasc_id=agasc_id, obs_status_override=obs_status_override, tstop=tstop
            )
            agasc_stats.append(agasc_stat)
            obs_stats.append(obs_stat)
            fails += obs_fail
        except mag_estimate.MagStatsException as e:
            msg = str(e)
            logger.debug(msg)
            fails.append(dict(e))
        except Exception as e:
            # transform Exception to MagStatsException for standard book keeping
            msg = f"Unexpected Error: {e}"
            logger.debug(msg)
            fails.append(
                dict(mag_estimate.MagStatsException(agasc_id=agasc_id, msg=msg))
            )
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type is not None:
                trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
                for level in trace:
                    for line in level.splitlines():
                        logger.debug(line)
                    logger.debug("")

    bar.close()
    logger.debug("-" * 80)

    try:
        agasc_stats = Table(agasc_stats) if agasc_stats else None
        obs_stats = vstack(obs_stats) if obs_stats else None
    except Exception as e:
        agasc_stats = None
        obs_stats = None
        # transform Exception to MagStatsException for standard book keeping
        fails.append(
            dict(
                mag_estimate.MagStatsException(
                    msg=f"Exception at end of get_agasc_id_stats: {str(e)}"
                )
            )
        )
    return obs_stats, agasc_stats, fails


def get_agasc_id_stats_pool(
    agasc_ids, obs_status_override=None, batch_size=100, tstop=None, no_progress=None
):
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

    from astropy.table import Table, vstack

    if obs_status_override is None:
        obs_status_override = {}

    jobs = []
    args = []
    finished = 0
    logger.info(f"Processing {batch_size} stars per job")
    for i in range(0, len(agasc_ids), batch_size):
        args.append(agasc_ids[i : i + batch_size])
    with Pool() as pool:
        for arg in args:
            jobs.append(
                pool.apply_async(
                    get_agasc_id_stats, [arg, obs_status_override, tstop, True]
                )
            )
        bar = tqdm(total=len(jobs), desc="progress", disable=no_progress, unit="job")
        while finished < len(jobs):
            finished = sum([f.ready() for f in jobs])
            if finished - bar.n:
                bar.update(finished - bar.n)
            time.sleep(1)
        bar.close()

    fails = []
    for arg, job in zip(args, jobs):
        if job.successful():
            continue
        try:
            job.get()
        except Exception as e:
            for agasc_id in arg:
                fails.append(
                    dict(
                        mag_estimate.MagStatsException(
                            agasc_id=agasc_id, msg=f"Failed job: {e}"
                        )
                    )
                )

    results = [job.get() for job in jobs if job.successful()]

    obs_stats = [r[0] for r in results if r[0] is not None]
    agasc_stats = [r[1] for r in results if r[1] is not None]
    obs_stats = vstack(obs_stats) if obs_stats else Table()
    agasc_stats = vstack(agasc_stats) if agasc_stats else Table()
    fails += sum([r[2] for r in results], [])

    return obs_stats, agasc_stats, fails


def _update_table(table_old, table_new, keys):
    # checking names, because actual types change upon saving in fits format
    if set(table_old.colnames) != set(table_new.colnames):
        raise Exception(
            "Tables have different columns:"
            f"\n  {table_old.colnames}"
            f"\n  {table_new.colnames}"
        )
    table_old = table_old.copy()
    new_row = np.ones(len(table_new), dtype=bool)
    _, i_new, i_old = np.intersect1d(
        table_new[keys].as_array(), table_old[keys].as_array(), return_indices=True
    )
    new_row[i_new] = False
    columns = table_old.as_array().dtype.names
    table_old[i_old] = table_new[i_new][columns]
    return table.vstack([table_old, table_new[new_row][columns]])


def update_mag_stats(obs_stats, agasc_stats, fails, outdir="."):
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
        filename = outdir / "mag_stats_agasc.fits"
        logger.debug(f"Updating {filename}")
        if filename.exists():
            agasc_stats = _update_table(
                table.Table.read(filename), agasc_stats, keys=["agasc_id"]
            )
            os.remove(filename)
        for column in agasc_stats.colnames:
            if column in mag_estimate.AGASC_ID_STATS_INFO:
                agasc_stats[column].description = mag_estimate.AGASC_ID_STATS_INFO[
                    column
                ]
        agasc_stats.write(filename)
    if obs_stats is not None and len(obs_stats):
        filename = outdir / "mag_stats_obsid.fits"
        logger.debug(f"Updating {filename}")
        if filename.exists():
            obs_stats = _update_table(
                table.Table.read(filename), obs_stats, keys=["agasc_id", "obsid"]
            )
            os.remove(filename)
        for column in obs_stats.colnames:
            if column in mag_estimate.OBS_STATS_INFO:
                obs_stats[column].description = mag_estimate.OBS_STATS_INFO[column]
        obs_stats.write(filename)
    if len(fails):
        filename = outdir / "mag_stats_fails.pkl"
        logger.info(f"A summary of all failures is saved in {filename}")
        # logger.debug(f'Updating {filename}')
        with open(filename, "wb") as out:
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
    if agasc_stats is None or len(agasc_stats) == 0:
        return [], []

    if include_all:
        outliers_new = agasc_stats[(agasc_stats["n_obsids_ok"] > 0)]
    else:
        outliers_new = agasc_stats[
            (agasc_stats["n_obsids_ok"] > 0)
            & (
                agasc_stats["selected_atol"]
                | agasc_stats["selected_rtol"]
                | agasc_stats["selected_color"]
                | agasc_stats["selected_mag_aca_err"]
            )
        ]
    outliers_new["mag_aca"] = outliers_new["mag_obs"]
    outliers_new["mag_aca_err"] = outliers_new["mag_obs_err"]

    outliers_new = outliers_new[MAGS_DTYPE.names].as_array()
    if outliers_new.dtype != MAGS_DTYPE:
        outliers_new = outliers_new.astype(MAGS_DTYPE)

    outliers = None
    new_stars = None
    updated_stars = None
    if filename.exists():
        # I could do what follows directly in place, but the table is not that large.
        with tables.File(filename, "r") as h5:
            if "mags" in h5.root:
                outliers_current = h5.root.mags[:]
                # find the indices of agasc_ids in both current and new lists
                _, i_new, i_cur = np.intersect1d(
                    outliers_new["agasc_id"],
                    outliers_current["agasc_id"],
                    return_indices=True,
                )
                current = outliers_current[i_cur]
                new = outliers_new[i_new]

                # from those, find the ones which differ in last observation time
                i_cur = i_cur[current["last_obs_time"] != new["last_obs_time"]]
                i_new = i_new[current["last_obs_time"] != new["last_obs_time"]]
                # overwrite current values with new values (and calculate diff to return)
                updated_stars = np.zeros(len(outliers_new[i_new]), dtype=MAGS_DTYPE)
                updated_stars["mag_aca"] = (
                    outliers_new[i_new]["mag_aca"] - outliers_current[i_cur]["mag_aca"]
                )
                updated_stars["mag_aca_err"] = (
                    outliers_new[i_new]["mag_aca_err"]
                    - outliers_current[i_cur]["mag_aca_err"]
                )
                updated_stars["agasc_id"] = outliers_new[i_new]["agasc_id"]
                outliers_current[i_cur] = outliers_new[i_new]

                # find agasc_ids in new list but not in current list
                new_stars = ~np.in1d(
                    outliers_new["agasc_id"], outliers_current["agasc_id"]
                )
                # and add them to the current list
                outliers_current = np.concatenate(
                    [outliers_current, outliers_new[new_stars]]
                )
                outliers = np.sort(outliers_current)

                new_stars = outliers_new[new_stars]["agasc_id"]

    if outliers is None:
        logger.warning('Creating new "mags" table')
        outliers = outliers_new
        new_stars = outliers_new["agasc_id"]
        updated_stars = np.array([], dtype=MAGS_DTYPE)

    mode = "r+" if filename.exists() else "w"
    with tables.File(filename, mode) as h5:
        if "mags" in h5.root:
            h5.remove_node("/mags")
        h5.create_table("/", "mags", outliers)
    save_version(filename, "mags")

    return new_stars, updated_stars


def write_obs_status_yaml(obs_stats=None, fails=(), filename=None):
    obs = []
    if obs_stats and len(obs_stats):
        obs_stats = obs_stats[~obs_stats["obs_ok"]]
        mp_starcat_times = np.unique(obs_stats["mp_starcat_time"])
        for mp_starcat_time in mp_starcat_times:
            rows = obs_stats[obs_stats["mp_starcat_time"] == mp_starcat_time]
            rows.sort(keys="agasc_id")
            obs.append(
                {
                    "mp_starcat_time": mp_starcat_time,
                    "obsid": obs_stats["obsid"],
                    "agasc_id": list(rows["agasc_id"]),
                    "status": 1,
                    "comments": obs_stats["comment"],
                }
            )
    for fail in fails:
        if fail["agasc_id"] is None or fail["mp_starcat_time"] is None:
            continue
        mp_starcat_times = (
            fail["mp_starcat_time"]
            if isinstance(fail["mp_starcat_time"], list)
            else [fail["mp_starcat_time"]]
        )
        agasc_id = fail["agasc_id"]
        for mp_starcat_time in mp_starcat_times:
            obs.append(
                {
                    "mp_starcat_time": mp_starcat_time,
                    "obsid": fail["obsid"],
                    "agasc_id": [agasc_id],
                    "status": 1,
                    "comments": fail["msg"],
                }
            )
    if len(obs) == 0:
        if filename and filename.exists():
            filename.unlink()
        return

    agasc_ids = []
    for o in obs:
        cat = get_starcheck_catalog(o["obsid"])
        if cat:
            cat = cat["cat"]
            maxmags = dict(zip(cat["id"], cat["maxmag"]))
            agasc_ids += [
                (agasc_id, maxmags.get(agasc_id, -1)) for agasc_id in o["agasc_id"]
            ]
        else:
            agasc_ids += [(agasc_id, -1) for agasc_id in obs["agasc_id"]]

    agasc_ids = dict(sorted(agasc_ids))

    yaml_template = """\
# See https://sot.github.io/agasc/supplement.html for detailed guidance.
#
# After reviewing and updating this file, run as `aca`:
#   agasc-supplement-tasks disposition
# Check the diffs at:
#   https://cxc.cfa.harvard.edu/mta/ASPECT/agasc/supplement/agasc_supplement_diff.html
# After reviewing the post-disposition email with diffs, run as `aca`:
#   agasc-supplement-tasks schedule-promotion
#
# Replace BAD_STAR_CODE with one of the following bad star codes:
#  9: Bad star added based on automated magnitude processing. Magnitude is a lower bound, set to MAXMAG.
# 10: Bad star added manually due to bad position (bad catalog position or high or missing proper motion)
# 11: Bad star added based on automated magnitude processing. Magnitude is an upper bound.
# 12: Bad star added based on automated magnitude processing. Magnitude set manually.
# 13: Bad star added based on automated magnitude processing. Star shows variability not captured in VAR.
# See also: https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually
bad:
  {%- for agasc_id, maxmag in agasc_ids.items() %}
  {{ agasc_id }}: BAD_STAR_CODE
  {%- endfor %}
#
# Mags below assume that the star is not detected and uses the star max mag.
# Edit accordingly if this is not the case.
mags:
  {%- for agasc_id, maxmag in agasc_ids.items() %}
  - agasc_id: {{ agasc_id }}
    mag_aca: {{ maxmag }}
    mag_aca_err: 0.1
  {%- endfor %}
#
# status: 0 = star-obs is OK and mag data should be used => REMOVE the mags entry
#         1 = star-obs is bad/unreliable => use mags values above
# comments: see the Rubric in https://sot.github.io/agasc/supplement.html
#   Most commonly:
#     "Never acquired" (status=1)
#     "Almost never acquired" (status=0 or 1, depending on data quality)
#     "Faint star" (status=0)
obs:
  {%- for obs in observations %}
  - mp_starcat_time: {{ obs.mp_starcat_time }}
    obsid: {{ obs.obsid }}
    status: {{ obs.status }}
    agasc_id: [{% for agasc_id in obs.agasc_id -%}
                  {{ agasc_id }}{%- if not loop.last %}, {% endif -%}
               {%- endfor -%}]
    comments: {{ obs.comments }}
  {%- endfor %}
"""  # noqa: E501
    tpl = jinja2.Template(yaml_template)
    result = tpl.render(observations=obs, agasc_ids=agasc_ids)
    if filename:
        with open(filename, "w") as fh:
            fh.write(result)
    return result


def do(
    start,
    stop,
    output_dir,
    agasc_ids=None,
    report=False,
    reports_dir=None,
    report_date=None,
    multi_process=False,
    include_bad=False,
    dry_run=False,
    no_progress=None,
    email="",
):
    """

    :param start: cxotime.CxoTime
        Start time. Only stars with at least one observation between start/stop are considered.
    :param stop: cxotime.CxoTime
        Stop time. Only stars with at least one observation between start/stop are considered.
    :param output_dir: pathlib.Path
        Directory where to place all output.
    :param agasc_ids: list
        List of AGASC IDs. Otional. If not given, all observations within start/stop are used.
    :param report: bool
        Generate an HTML report.
    :param reports_dir: pathlib.Path
        Directory where to write reports.
    :param report_date: cxotime.CxoTime
        The report date (report_date.date[:8] will be the report directory name)
    :param multi_process: bool
        Run on mulyiple processes.
    :param include_bad: bool
        Consider stars that are in the 'bad' supplement table.
    :param dry_run: bool
        Only parse options and not actually run the magnitude estimates
    :param no_progress: bool
        Hide progress bar
    :param email: str
    :return:
    """
    # PyTables is not really unicode-compatible, but python 3 is basically unicode.
    # For our purposes, PyTables works. It would fail with characters that can not be written
    # as ascii. It displays a warning which I want to avoid:
    warnings.filterwarnings("ignore", category=tables.exceptions.FlavorWarning)

    filename = output_dir / "agasc_supplement.h5"

    if multi_process:
        get_stats = partial(get_agasc_id_stats_pool, batch_size=10)
    else:
        get_stats = get_agasc_id_stats

    skip = True
    if agasc_ids is None:
        obs_in_time = (star_obs_catalogs.STARS_OBS["mp_starcat_time"] >= start) & (
            star_obs_catalogs.STARS_OBS["mp_starcat_time"] <= stop
        )
        agasc_ids = sorted(star_obs_catalogs.STARS_OBS[obs_in_time]["agasc_id"])
    else:
        agasc_ids = np.intersect1d(agasc_ids, star_obs_catalogs.STARS_OBS["agasc_id"])
        skip = False

    agasc_ids = np.unique(agasc_ids)
    stars_obs = star_obs_catalogs.STARS_OBS[
        np.in1d(star_obs_catalogs.STARS_OBS["agasc_id"], agasc_ids)
    ]

    # if supplement exists:
    # - drop bad stars
    # - get OBS status override
    # - get the latest observation for each agasc_id,
    # - find the ones already in the supplement
    # - include only the ones with supplement.last_obs_time < than stars_obs.mp_starcat_time
    obs_status_override = {}
    if filename.exists():
        with tables.File(filename, "r") as h5:
            if not include_bad and "bad" in h5.root:
                logger.info("Excluding bad stars")
                stars_obs = stars_obs[
                    ~np.in1d(stars_obs["agasc_id"], h5.root.bad[:]["agasc_id"])
                ]

            if "obs" in h5.root:
                obs_status_override = table.Table(h5.root.obs[:])
                obs_status_override.convert_bytestring_to_unicode()
                obs_status_override = {
                    (r["mp_starcat_time"], r["agasc_id"]): {
                        "status": r["status"],
                        "comments": r["comments"],
                    }
                    for r in obs_status_override
                }
            if "mags" in h5.root and len(stars_obs):
                outliers_current = h5.root.mags[:]
                times = (
                    stars_obs[["agasc_id", "mp_starcat_time"]]
                    .group_by("agasc_id")
                    .groups.aggregate(lambda d: np.max(CxoTime(d)).date)
                )
                if len(outliers_current):
                    times = table.join(
                        times,
                        table.Table(outliers_current[["agasc_id", "last_obs_time"]]),
                        join_type="left",
                    )
                else:
                    times["last_obs_time"] = table.MaskedColumn(
                        np.zeros(len(times), dtype=h5.root.mags.dtype["last_obs_time"]),
                        mask=np.ones(len(times), dtype=bool),
                    )
                if skip:
                    if hasattr(times["last_obs_time"], "mask"):
                        # the mask exists if there are stars in stars_obs
                        # that are not in outliers_current
                        update = times["last_obs_time"].mask | (
                            (~times["last_obs_time"].mask)
                            & (
                                CxoTime(times["mp_starcat_time"]).cxcsec
                                > times["last_obs_time"]
                            ).data
                        )
                    else:
                        update = (
                            CxoTime(times["mp_starcat_time"]).cxcsec
                            > times["last_obs_time"]
                        )

                    stars_obs = stars_obs[
                        np.in1d(stars_obs["agasc_id"], times[update]["agasc_id"])
                    ]
                    agasc_ids = np.sort(np.unique(stars_obs["agasc_id"]))
                    if len(update) - np.sum(update):
                        logger.info(
                            f"Skipping {len(update) - np.sum(update)} "
                            "stars already in the supplement"
                        )

    if len(stars_obs) == 0:
        logger.info("There are no new observations to process")
        return

    # do the processing
    logger.info(f"Will process {len(agasc_ids)} stars on {len(stars_obs)} observations")
    logger.info(f"from {start} to {stop}")

    obs_times = CxoTime(star_obs_catalogs.STARS_OBS["mp_starcat_time"])
    latest = np.sort(
        np.unique(
            star_obs_catalogs.STARS_OBS[["mp_starcat_time", "obsid"]][
                obs_times > obs_times.max() - 1 * u.day
            ]
        )
    )[-10:]
    logger.info("latest observations:")
    for row in latest:
        logger.info(
            f"  mp_starcat_time: {row['mp_starcat_time']}, OBSID {row['obsid']}"
        )
    if dry_run:
        return

    obs_stats, agasc_stats, fails = get_stats(
        agasc_ids,
        tstop=stop,
        obs_status_override=obs_status_override,
        no_progress=no_progress,
    )

    failed_global = [f for f in fails if not f["agasc_id"] and not f["obsid"]]
    failed_stars = [f for f in fails if f["agasc_id"] and not f["obsid"]]
    failed_obs = [f for f in fails if f["obsid"]]
    msg = (
        "Got:\n"
        f"  {0 if obs_stats is None else len(obs_stats)} OBSIDs,"
        f"  {0 if agasc_stats is None else len(agasc_stats)} stars,"
    )
    if failed_obs:
        msg += f"  {len(failed_obs)} failed observations,"
    if failed_stars:
        msg += f"  {len(failed_stars)} failed stars,"
    if failed_global:
        msg += f"  {len(failed_global)} global errors"
    logger.info(msg)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    update_mag_stats(obs_stats, agasc_stats, fails, output_dir)

    obs_status_file = output_dir / "obs_status.yml"
    try:
        write_obs_status_yaml(
            [], fails=failed_obs + failed_stars, filename=obs_status_file
        )
    except Exception as e:
        logger.warning(f"Failed to write {obs_status_file}: {e}")

    new_stars, updated_stars = update_supplement(agasc_stats, filename=filename)
    logger.info(f"  {len(new_stars)} new stars, {len(updated_stars)} updated stars")

    if agasc_stats is not None and len(agasc_stats):
        if email:
            try:
                bad_obs = (
                    (obs_stats["mp_starcat_time"] >= start)
                    & (obs_stats["mp_starcat_time"] < stop)
                    & ~obs_stats["obs_ok"]
                )
                if np.any(bad_obs):
                    msr.email_bad_obs_report(obs_stats[bad_obs], to=email)
            except Exception as e:
                logger.error(f"Error sending email to {email}: {e}")

    if report and len(agasc_stats):
        if report_date is None:
            report_dir = reports_dir
            report_data_file = report_dir / "report_data.pkl"
            nav_links = None
            report_date = CxoTime.now()
        else:
            report_dir = reports_dir / f"{report_date.date[:8]}"
            report_data_file = report_dir / f"report_data_{report_date.date[:8]}.pkl"

            week = time.TimeDelta(7 * u.day)
            nav_links = {
                "previous": f"../{(report_date - week).date[:8]}/index.html",
                "up": "..",
                "next": f"../{(report_date + week).date[:8]}/index.html",
            }

        # If the report data file exists, the arguments for the report from the file are
        # modified according to the current run. Otherwise, they are created from scratch.
        if report_data_file.exists():
            with open(report_data_file, "rb") as fh:
                report_data = pickle.load(fh)
            logger.info(f"Loading existing report data from {report_data_file}")
            multi_star_html_args = report_data["args"]

            # arguments for the report are modified here
            # merge fails:
            # - from previous run, take fails that were not run just now
            # - add current fails
            multi_star_html_args["fails"] = fails
            multi_star_html_args["no_progress"] = no_progress

        else:
            sections = [
                {"id": "new_stars", "title": "New Stars", "stars": new_stars},
                {
                    "id": "updated_stars",
                    "title": "Updated Stars",
                    "stars": updated_stars["agasc_id"] if len(updated_stars) else [],
                },
                {
                    "id": "other_stars",
                    "title": "Other Stars",
                    "stars": list(
                        agasc_stats["agasc_id"][
                            ~np.in1d(agasc_stats["agasc_id"], new_stars)
                            & ~np.in1d(
                                agasc_stats["agasc_id"], updated_stars["agasc_id"]
                            )
                        ]
                    ),
                },
            ]

            multi_star_html_args = {
                "filename": "index.html",
                "sections": sections,
                "updated_stars": updated_stars,
                "fails": fails,
                "report_date": report_date.date,
                "tstart": start,
                "tstop": stop,
                "nav_links": nav_links,
                "include_all_stars": False,
                "no_progress": no_progress,
            }

        try:
            report = msr.MagEstimateReport(
                agasc_stats=output_dir / "mag_stats_agasc.fits",
                obs_stats=output_dir / "mag_stats_obsid.fits",
                directory=report_dir,
            )
            report.multi_star_html(**multi_star_html_args)
            latest = reports_dir / "latest"
            if os.path.lexists(latest):
                logger.debug('Removing existing "latest" symlink')
                latest.unlink()
            logger.debug('Creating "latest" symlink')
            latest.symlink_to(report_dir.absolute())
        except Exception as e:
            report_dir = output_dir
            logger.error(f"Error when creating report: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type is not None:
                trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
                for level in trace:
                    for line in level.splitlines():
                        logger.debug(line)
                    logger.debug("")
        finally:
            report_data_file = report_dir / report_data_file.name
            if not report_dir.exists():
                report_dir.mkdir(parents=True)
            report_data = {"args": multi_star_html_args, "directory": report_dir}
            with open(report_data_file, "wb") as fh:
                pickle.dump(report_data, fh)
            logger.info(f"Report data saved in {report_data_file}")
    elif len(agasc_stats) == 0:
        logger.info("Nothing to report (no stars)")

    now = datetime.datetime.now()
    logger.info(f"done at {now}")
