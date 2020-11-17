#!/usr/bin/env python

"""
Update Magnitude Statistics.
"""

import argparse
import logging
from agasc.supplement.magnitudes import update_mag_supplement
import pathlib
import os


def parser():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument('--agasc-id-file', help='File containing a list of AGASC IDs, one per line.')
    parse.add_argument('--start',
                       help='Include only stars observed after this time.'
                            ' CxoTime-compatible time stamp.'
                            ' Default: now - 14 days.')
    parse.add_argument('--stop',
                       help='Include only stars observed before this time.'
                            ' CxoTime-compatible time stamp.'
                            ' Default: now.')
    parse.add_argument('--whole-history',
                       help='Include all star observations.')
    parse.add_argument('--obs-status-override', help='YAML file with observation status.')
    parse.add_argument('--obs', help='Observation ID for status override.')
    parse.add_argument('--agasc-id', help='AGASC ID for status override.')
    parse.add_argument('--status', help='Status to override.')
    parse.add_argument('--comments', help='Comments for status override.', default='')
    parse.add_argument('--email', help='Email to report errors.')
    parse.add_argument('--report',
                       help='Generate HTML report for the period covered. Default: False',
                       action='store_true', default=False)
    parse.add_argument('--output-dir',
                       help='Directory where to place the supplement. Default: .',
                       default='.')
    parse.add_argument('--reports-dir',
                       help='Directory where to place reports.'
                            ' Default: $SKA/www/ASPECT/agasc/supplement_reports/weekly.',
                       default='$SKA/www/ASPECT/agasc/supplement_reports/weekly')
    parse.add_argument('--multi-process',
                       help="Use multi-processing to accelerate run.",
                       action='store_true', default=False)
    parse.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    return parse


def main():
    the_parser = parser()
    args = the_parser.parse_args()

    args.output_dir = pathlib.Path(os.path.expandvars(args.output_dir))
    args.reports_dir = pathlib.Path(os.path.expandvars(args.reports_dir))

    logging.basicConfig(level=args.log_level.upper(),
                        format='%(message)s')

    if (args.obs and not args.status) or (not args.obs and args.status):
        logging.error('To override status, both --obs and --status options are needed.')
        the_parser.exit(1)

    update_mag_supplement.do(args)


if __name__ == '__main__':
    import warnings
    import numpy as np
    np.seterr(all='ignore')
    warnings.simplefilter('ignore', UserWarning)

    main()
