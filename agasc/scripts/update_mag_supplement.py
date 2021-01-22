#!/usr/bin/env python

"""
Update Magnitude Statistics.


"""

import argparse
import logging
from agasc.supplement.magnitudes import star_obs_catalogs
from agasc.supplement.magnitudes import update_mag_supplement
from agasc.scripts.add_bad_star import add_bad_star
from agasc.scripts import update_obs_status
import pathlib
import os
import pyyaks


def parser():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument('--start',
                       help='Include only stars observed after this time.'
                            ' CxoTime-compatible time stamp.'
                            ' Default: now - 14 days.')
    parse.add_argument('--stop',
                       help='Include only stars observed before this time.'
                            ' CxoTime-compatible time stamp.'
                            ' Default: now.')
    parse.add_argument('--whole-history',
                       help='Include all star observations and ignore --start/stop.',
                       action='store_true', default=False)
    parse.add_argument('--agasc-id-file',
                       help='Include only observations of stars whose AGASC IDs are specified '
                            'in this file, one per line.')
    parse.add_argument('--output-dir',
                       help='Directory where agasc_supplement.h5 is located.'
                            'Other output is placed here as well. Default: .',
                       default='.')
    parse.add_argument('--include-bad',
                       help='Do not exclude "bad" stars from magnitude estimates. Default: False',
                       action='store_true', default=False)
    status = parse.add_argument_group(
        'OBS/star status',
        'options to modify the "bads" and "obs" tables in AGASC supplement. '
        'Modifications to supplement happen before all magnitude estimates are made.'
    )
    status.add_argument('--obs-status-override',
                        help='YAML file with star/observation status. '
                             'More info at https://sot.github.io/agasc/supplement.html')
    status.add_argument('--obs', help='Observation ID for status override.')
    status.add_argument('--agasc-id', help='AGASC ID for status override.')
    status.add_argument('--status', help='Status to override.')
    status.add_argument('--bad-star', help='Bad star AGASC ID.',
                        default=[], action='append', type=int)
    status.add_argument('--comments', help='Comments for status override.', default='')
    report = parse.add_argument_group('Reporting')
    report.add_argument('--report',
                        help='Generate HTML report for the period covered. Default: False',
                        action='store_true', default=False)
    report.add_argument('--reports-dir',
                        help='Directory where to place reports.'
                             ' Default: <output_dir>/supplement_reports/weekly.')

    other = parse.add_argument_group('Other')
    other.add_argument('--multi-process',
                       help="Use multi-processing to accelerate run.",
                       action='store_true', default=False)
    other.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    other.add_argument("--dry-run",
                       action="store_true",
                       help="Dry run (no actual file or database updates)")
    return parse


def main():
    the_parser = parser()
    args = the_parser.parse_args()

    status_to_int = {'true': 1, 'false': 0, 'ok': 1, 'good': 1, 'bad': 0}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    args.output_dir = pathlib.Path(os.path.expandvars(args.output_dir))
    if args.reports_dir is None:
        args.reports_dir = args.output_dir / 'supplement_reports' / 'weekly'
    else:
        args.reports_dir = pathlib.Path(os.path.expandvars(args.reports_dir))

    pyyaks.logger.get_logger(
        name='agasc.supplement',
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s"
    )

    if (args.obs and not args.status) or (not args.obs and args.status):
        logging.error('To override OBS status, both --obs and --status options are needed.')
        the_parser.exit(1)

    star_obs_catalogs.load(args.stop)

    status_override = update_obs_status.parse_obs_status_args(
        filename=args.obs_status_override, **vars(args))

    # set the list of AGASC IDs from file if specified. If not, it will include all.
    agasc_ids = []
    if args.agasc_id_file:
        with open(args.agasc_id_file, 'r') as f:
            agasc_ids = [int(l.strip()) for l in f.readlines()]
    agasc_ids += [o[1] for o in status_override['obs']]

    if status_override['obs']:
        update_obs_status.update_obs_status(
            args.output_dir / 'agasc_supplement.h5', status_override['obs'], dry_run=args.dry_run
        )

    if status_override['bad']:
        bad_star_ids, bad_star_source = zip(*status_override['bad'].items())
        add_bad_star(bad_star_ids,
                     bad_star_source,
                     args.output_dir / 'agasc_supplement.h5',
                     dry_run=args.dry_run)

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
