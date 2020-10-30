#!/usr/bin/env python

"""
Update Magnitude Statistics.
"""

import argparse
import logging
from agasc.supplement.magnitudes import update_mag_supplement


def parser():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument('--agasc-id-file', help='File containing a list of AGASC IDs, one per line.')
    parse.add_argument('--start',
                       help='Time to start processing new observations.'
                            ' CxoTime-compatible time stamp.')
    parse.add_argument('--stop',
                       help='Time to stop processing new observations.'
                            ' CxoTime-compatible time stamp.')
    parse.add_argument('--obs-status-override', help='YAML file with observation status.')
    parse.add_argument('--obs', help='Observation ID for status override.')
    parse.add_argument('--agasc-id', help='AGASC ID for status override.')
    parse.add_argument('--status', help='Status to override.')
    parse.add_argument('--comments', help='Comments for status override.', default='')
    parse.add_argument('--email', help='Email to report errors.')
    parse.add_argument('--report', help='Generate HTML report for the period covered',
                       action='store_true', default=False)
    parse.add_argument('--multi-process', help="Use multi-processing to accelerate run",
                       action='store_true', default=False)
    parse.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    return parse


def main():
    the_parser = parser()
    args = the_parser.parse_args()

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