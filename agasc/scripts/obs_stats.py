#!/usr/bin/env python

"""
Observation Statistics.


"""

import argparse
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat

import numpy as np
import yaml
from astropy.table import Table, vstack
from cxotime import CxoTime
from cxotime import units as u
from ska_helpers import logging as ska_logging
from tqdm import tqdm

from agasc import agasc
from agasc.supplement.magnitudes import mag_estimate, star_obs_catalogs


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--start",
        help=(
            "Include only stars observed after this time."
            " CxoTime-compatible time stamp."
            " Default: now - 30 days."
        ),
    )
    parser.add_argument(
        "--stop",
        help=(
            "Include only stars observed before this time."
            " CxoTime-compatible time stamp."
            " Default: now."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=("Directory where to write the result. Default: ."),
        default=".",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--multiprocessing",
        help="Use multiprocessing to speed up the processing of observations.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save-call-args",
        help=(
            "Save the input arguments to a YAML file in the output directory."
            " The file name is call_args.yml or call_args.N.yml if the former already exists."
        ),
        action="store_true",
        default=False,
    )
    return parser


def get_args():
    logger = ska_logging.basic_logger(
        name="agasc.supplement",
        level="WARNING",
        format="%(asctime)s %(message)s",
    )

    the_parser = get_parser()
    args = the_parser.parse_args()
    logger.setLevel(args.log_level.upper())

    args.output_dir = Path(os.path.expandvars(args.output_dir))

    # set start/stop times
    args.stop = CxoTime(args.stop).date if args.stop else CxoTime.now().date
    args.start = (
        CxoTime(args.start).date if args.start else (CxoTime.now() - 30 * u.day).date
    )

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    # save call args just in case
    if args.save_call_args:
        args_log_file = get_next_file_name(args.output_dir / "call_args.yml")
        yaml_args = {
            k: str(v) if issubclass(type(v), Path) else v for k, v in vars(args).items()
        }
        logger.info(f"Writing input arguments to {args_log_file}")
        with open(args_log_file, "w") as fh:
            yaml.dump(yaml_args, fh)

        logger.info("Input arguments")
        for line in pformat(yaml_args).split("\n"):
            logger.info(line.rstrip())

    return {
        "output_dir": args.output_dir,
        "start": args.start,
        "stop": args.stop,
        "multiprocessing": args.multiprocessing,
    }


def get_next_file_name(file_name):
    if not file_name.exists():
        return file_name
    i = 1
    while True:
        new_file_name = file_name.with_suffix(f".{i}{file_name.suffix}")
        if not new_file_name.exists():
            return new_file_name
        i += 1


def get_multi_obs_stats(star_obs, obs_status_override):
    telem = mag_estimate.get_telemetry_by_observations(
        star_obs, ignore_exceptions=True, as_table=False
    )

    # Only keep telemetry from the first 2000 seconds of each observation
    for tel in telem:
        if "times" not in tel or len(tel["times"]) == 0:
            continue
        t0 = tel["times"][0]
        sel = (tel["times"] - t0) < 2000
        for k in tel:
            tel[k] = tel[k][sel]

    obs_stats, failures = mag_estimate.get_multi_obs_stats(
        star_obs, telem=telem, obs_status_override=obs_status_override
    )
    return obs_stats, failures


def get_multi_obs_stats_pool(
    star_obs, obs_status_override, batch_size=20, no_progress=None
):
    """
    Call update_mag_stats.get_agasc_id_stats multiple times using a multiprocessing.Pool

    :param star_obs: Table
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
    logger = logging.getLogger("agasc.supplement")

    jobs = []
    args = []
    finished = 0
    logger.info(f"Processing {batch_size} observations per job")
    for i in range(0, len(star_obs), batch_size):
        args.append(star_obs[i : i + batch_size])

    with Pool() as pool:
        for arg in args:
            jobs.append(
                pool.apply_async(get_multi_obs_stats, [arg, obs_status_override])
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
            for obs in arg:
                fails.append(
                    dict(
                        mag_estimate.MagStatsException(
                            agasc_id=obs["agasc_id"],
                            obsid=obs["obsid"],
                            msg=f"Failed job: {e}",
                        )
                    )
                )

    results = [job.get() for job in jobs if job.successful()]

    obs_stats = [r[0] for r in results if r[0] is not None]
    obs_stats = vstack(obs_stats) if obs_stats else Table()
    fails += sum([r[1] for r in results], [])

    return obs_stats, fails


def main():
    args = get_args()

    star_obs_catalogs.load(args["stop"])

    obs_status_override = {
        (r["mp_starcat_time"], r["agasc_id"]): {
            "status": r["status"],
            "comments": r["comments"],
        }
        for r in agasc.get_supplement_table("obs")
    }

    obs_in_time = (star_obs_catalogs.STARS_OBS["mp_starcat_time"] >= args["start"]) & (
        star_obs_catalogs.STARS_OBS["mp_starcat_time"] <= args["stop"]
    )
    star_obs = star_obs_catalogs.STARS_OBS[obs_in_time]

    if args["multiprocessing"]:
        obs_stats, failures = get_multi_obs_stats_pool(star_obs, obs_status_override)
    else:
        obs_stats, failures = get_multi_obs_stats(star_obs, obs_status_override)

    if len(obs_stats) > 0:
        print(f"Successfully processed {len(obs_stats)} observations.")
        obs_stats.sort(["mp_starcat_time", "slot"])
        obs_stats.write(args["output_dir"] / "obs_stats.fits", overwrite=True)
    else:
        print("No observations processed successfully.")

    if failures:
        # this cleans it up for YAML dumping, converting numpy types to native Python types where
        # possible.
        failures = [
            {
                k: (v.item() if isinstance(v, np.generic) and hasattr(v, "item") else v)
                for k, v in d.items()
            }
            for d in failures
        ]
        with open(args["output_dir"] / "obs_stats_failures.yml", "w") as fh:
            yaml.dump(failures, fh)


if __name__ == "__main__":
    main()
