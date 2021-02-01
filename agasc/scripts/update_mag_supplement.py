#!/usr/bin/env python

"""
Update Magnitude Statistics.


"""
import os
from pathlib import Path
import argparse
import logging
import pyyaks.logger
from agasc.supplement.magnitudes import star_obs_catalogs
from agasc.supplement.magnitudes import update_mag_supplement
from agasc.scripts import update_obs_status


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[update_obs_status.get_obs_status_parser()]
    )
    parser.add_argument('--start',
                        help='Include only stars observed after this time.'
                             ' CxoTime-compatible time stamp.'
                             ' Default: now - 14 days.')
    parser.add_argument('--stop',
                        help='Include only stars observed before this time.'
                             ' CxoTime-compatible time stamp.'
                             ' Default: now.')
    parser.add_argument('--whole-history',
                        help='Include all star observations and ignore --start/stop.',
                        action='store_true', default=False)
    parser.add_argument('--agasc-id-file',
                        help='Include only observations of stars whose AGASC IDs are specified '
                             'in this file, one per line.')
    parser.add_argument('--output-dir',
                        help='Directory where agasc_supplement.h5 is located.'
                             'Other output is placed here as well. Default: .',
                        default='.')
    parser.add_argument('--include-bad',
                        help='Do not exclude "bad" stars from magnitude estimates. Default: False',
                        action='store_true', default=False)
    report = parser.add_argument_group('Reporting')
    report.add_argument('--report',
                        help='Generate HTML report for the period covered. Default: False',
                        action='store_true', default=False)
    report.add_argument('--reports-dir',
                        help='Directory where to place reports.'
                             ' Default: <output_dir>/supplement_reports/weekly.')

    other = parser.add_argument_group('Other')
    other.add_argument('--multi-process',
                       help="Use multi-processing to accelerate run.",
                       action='store_true', default=False)
    other.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    other.add_argument("--dry-run",
                       action="store_true",
                       help="Dry run (no actual file or database updates)")
    return parser


def main():
    logger = logging.getLogger('agasc.supplement')
    the_parser = get_parser()
    args = the_parser.parse_args()

    status_to_int = {'true': 1, 'false': 0, 'ok': 1, 'good': 1, 'bad': 0}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    args.output_dir = Path(os.path.expandvars(args.output_dir))
    if args.reports_dir is None:
        args.reports_dir = args.output_dir / 'supplement_reports' / 'weekly'
    else:
        args.reports_dir = Path(os.path.expandvars(args.reports_dir))

    if args.whole_history:
        if args.start or args.stop:
            logger.error('--whole-history argument is incompatible with --start/--stop arguments')
            the_parser.exit(1)
        args.start = None
        args.stop = None

    pyyaks.logger.get_logger(
        name='agasc.supplement',
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s"
    )

    if (args.obsid and not args.status) or (not args.obsid and args.status):
        logger.error('To override OBS status, both --obs and --status options are needed.')
        the_parser.exit(1)

    star_obs_catalogs.load(args.stop)

    # set the list of AGASC IDs from file if specified. If not, it will include all.
    agasc_ids = []
    if args.agasc_id_file:
        with open(args.agasc_id_file, 'r') as f:
            agasc_ids = [int(line.strip()) for line in f.readlines()]

    # update 'bad' and 'obs' tables in supplement
    agasc_ids += update_obs_status.update(args)

    update_mag_supplement.do(
        args.output_dir,
        args.reports_dir,
        agasc_ids if agasc_ids else None,
        args.multi_process,
        args.start,
        args.stop,
        args.whole_history,
        args.report,
        include_bad=args.include_bad,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    import warnings
    import numpy as np
    np.seterr(all='ignore')
    warnings.simplefilter('ignore', UserWarning)

    main()
