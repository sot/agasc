#!/usr/bin/env python

"""
Update Magnitude Statistics.


"""
import argparse
import logging
import os
from pathlib import Path
from pprint import pformat

import pyyaks.logger
import yaml
from cxotime import CxoTime
from cxotime import units as u

from agasc.scripts import update_supplement
from agasc.supplement.magnitudes import star_obs_catalogs, update_mag_supplement


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, parents=[update_supplement.get_obs_status_parser()]
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
        "--whole-history",
        help="Include all star observations and ignore --start/stop.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--agasc-id-file",
        help=(
            "Include only observations of stars whose AGASC IDs are specified "
            "in this file, one per line."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where agasc_supplement.h5 is located."
            "Other output is placed here as well. Default: ."
        ),
        default=".",
    )
    parser.add_argument(
        "--include-bad",
        help='Do not exclude "bad" stars from magnitude estimates. Default: False',
        action="store_true",
        default=False,
    )
    report = parser.add_argument_group("Reporting")
    report.add_argument(
        "--report",
        help="Generate HTML report for the period covered. Default: False",
        action="store_true",
        default=False,
    )
    report.add_argument(
        "--reports-dir",
        help=(
            "Directory where to place reports."
            " Default: <output_dir>/supplement_reports/weekly."
        ),
    )

    other = parser.add_argument_group("Other")
    other.add_argument(
        "--multi-process",
        help="Use multi-processing to accelerate run.",
        action="store_true",
        default=False,
    )
    other.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    other.add_argument(
        "--no-progress",
        dest="no_progress",
        help="Do not show a progress bar",
        action="store_true",
    )  # this has no default, it will be None.
    other.add_argument(
        "--args-file",
        help=(
            "YAML file with arguments to "
            "agasc.supplement.magnitudes.update_mag_supplement.do"
        ),
    )
    other.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual file or database updates)",
    )
    return parser


def get_args():
    logger = logging.getLogger("agasc.supplement")
    the_parser = get_parser()
    args = the_parser.parse_args()
    if args.args_file:
        args.args_file = Path(args.args_file)
        if not args.args_file.exists():
            logger.error(f"File does not exist: {args.args_file}")
            the_parser.exit(1)
        with open(args.args_file) as fh:
            file_args = yaml.load(fh, Loader=yaml.SafeLoader)
            the_parser.set_defaults(**file_args)
        args = the_parser.parse_args()

    status_to_int = {"true": 1, "false": 0, "ok": 1, "good": 1, "bad": 0}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    args.output_dir = Path(os.path.expandvars(args.output_dir))
    if args.reports_dir is None:
        args.reports_dir = args.output_dir / "supplement_reports" / "weekly"
    else:
        args.reports_dir = Path(os.path.expandvars(args.reports_dir))

    if args.whole_history:
        if args.start or args.stop:
            logger.error(
                "--whole-history argument is incompatible with --start/--stop arguments"
            )
            the_parser.exit(1)
        args.start = None
        args.stop = None

    pyyaks.logger.get_logger(
        name="agasc.supplement",
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s",
    )

    if ((args.obsid or args.mp_starcat_time) and not args.status) or (
        not (args.obsid or args.mp_starcat_time) and args.status
    ):
        logger.error(
            "To override OBS status, both --obs/mp-starcat-time and --status options"
            " are needed."
        )
        the_parser.exit(1)

    star_obs_catalogs.load(args.stop)

    if "agasc_ids" in file_args:
        agasc_ids = file_args["agasc_ids"]
    else:
        # set the list of AGASC IDs from file if specified. If not, it will include all.
        agasc_ids = []
        if args.agasc_id_file:
            with open(args.agasc_id_file, "r") as f:
                agasc_ids = [int(line.strip()) for line in f.readlines()]

        # update 'bad' and 'obs' tables in supplement
        agasc_ids += update_supplement.update(args)

    # set start/stop times
    if args.whole_history:
        if args.start or args.stop:
            raise ValueError("incompatible arguments: whole_history and start/stop")
        args.start = CxoTime(star_obs_catalogs.STARS_OBS["mp_starcat_time"]).min().date
        args.stop = CxoTime(star_obs_catalogs.STARS_OBS["mp_starcat_time"]).max().date
    else:
        args.stop = CxoTime(args.stop).date if args.stop else CxoTime.now().date
        args.start = (
            CxoTime(args.start).date
            if args.start
            else (CxoTime.now() - 30 * u.day).date
        )

    report_date = None
    if "report_date" in file_args:
        report_date = CxoTime(file_args["report_date"])
    elif args.report:
        report_date = CxoTime(args.stop)
        # the nominal date for reports is the first Monday after the stop date.
        # this is not perfect, because it needs to agree with nav_links in update_mag_supplement.do
        report_date += ((7 - report_date.datetime.weekday()) % 7) * u.day
        report_date = CxoTime(report_date.date[:8])

    args_log_file = args.output_dir / "call_args.yml"
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    # there must be a better way to do this...
    yaml_args = {
        k: str(v) if issubclass(type(v), Path) else v for k, v in vars(args).items()
    }
    yaml_args["report_date"] = report_date.date
    yaml_args["agasc_ids"] = agasc_ids
    with open(args_log_file, "w") as fh:
        yaml.dump(yaml_args, fh)

    logger.info("Input arguments")
    for line in pformat(yaml_args).split("\n"):
        logger.info(line.rstrip())

    return dict(
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        report_date=report_date,
        agasc_ids=agasc_ids if agasc_ids else None,
        multi_process=args.multi_process,
        start=args.start,
        stop=args.stop,
        report=args.report,
        include_bad=args.include_bad,
        dry_run=args.dry_run,
        no_progress=args.no_progress,
        args_log_file=args_log_file,
    )


def main():
    import kadi.commands

    kadi.commands.conf.commands_version = "1"

    args = get_args()
    args_log_file = args.pop("args_log_file")

    update_mag_supplement.do(**args)

    if (
        args["report"]
        and (args["reports_dir"] / f"{args['report_date'].date[:8]}").exists()
    ):
        args_log_file.replace(
            args["reports_dir"] / f"{args['report_date'].date[:8]}" / args_log_file.name
        )


if __name__ == "__main__":
    main()
