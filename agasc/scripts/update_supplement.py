#!/usr/bin/env python
"""
Update the 'bad' and 'obs' tables in the agasc supplement (agasc_supplement.h5).

This file is a supplement to the stable AGASC to inform star selection
and star catalog checking.

This script can add one or many bad stars to the bad star table, as well as observation status
information relevant for the magnitude estimation from acq data.
"""

import argparse
from pathlib import Path
import logging
from cxotime.cxotime import CxoTime
import yaml
import numpy as np
import pyyaks.logger

from agasc.supplement.magnitudes import star_obs_catalogs as cat
from agasc.supplement.utils import update_mags_table, update_obs_table, add_bad_star


logger = logging.getLogger('agasc.supplement')


def _parse_obs_status_file(filename):
    """
    Parse a yaml file and return a dictionary.

    The dictionary will be of the form: {'obs': [], 'bad': [], 'mags: []}

    :param filename:
    :return:
    """
    with open(filename) as fh:
        status = yaml.load(fh, Loader=yaml.SafeLoader)
    if 'obs' not in status:
        status['obs'] = []
    if 'bad' not in status:
        status['bad'] = []
    if 'mags' not in status:
        status['mags'] = []

    if hasattr(status['bad'], 'items'):
        status['bad'] = list(status['bad'].items())

    return status


def _sanitize_args(status):
    for star in status['mags']:
        if 'last_obs_time' in star:
            star['last_obs_time'] = CxoTime(star['last_obs_time']).cxcsec
        else:
            obs = cat.STARS_OBS[cat.STARS_OBS['agasc_id'] == star['agasc_id']]
            if len(obs) == 0:
                raise Exception(f"Can not guess last_obs_time for agasc_id={star['agasc_id']}")
            star['last_obs_time'] = CxoTime(obs['mp_starcat_time']).max().cxcsec

    for value in status['obs']:
        if cat.STARS_OBS is None:
            raise RuntimeError('Observation catalog is not initialized')
        if 'mp_starcat_time' in value:
            key = 'mp_starcat_time'
            rows = cat.STARS_OBS[cat.STARS_OBS['mp_starcat_time'] == value['mp_starcat_time']]
        elif 'obsid' in value:
            key = 'obsid'
            rows = cat.STARS_OBS[cat.STARS_OBS['obsid'] == value['obsid']]
        else:
            raise Exception('Need to specify mp_starcat_time or OBSID')
        if len(rows) == 0:
            raise Exception(
                f'Observation catalog has no observation with {key}={value[key]}'
            )

        if 'agasc_id' not in value:
            value['agasc_id'] = list(sorted(rows['agasc_id']))
        else:
            value['agasc_id'] = (list(np.atleast_1d(value['agasc_id'])))
        if 'comments' not in value:
            value['comments'] = ''
        if 'mp_starcat_time' not in value:
            value['mp_starcat_time'] = rows['mp_starcat_time'][0]
        if 'obsid' not in value:
            value['obsid'] = rows['obsid'][0]

        # sanity checks:
        # AGASC IDs are not checked to exist because a star could be observed without it being
        # in the starcheck catalog (a spoiler star), but the observation must exist and have
        # matching mp_starcat_time and OBSID
        if rows['obsid'][0] != value['obsid']:
            raise Exception(f'inconsistent observation spec {value}')
        if rows['mp_starcat_time'][0] != value['mp_starcat_time']:
            raise Exception(f'inconsistent observation spec {value}')

    rows = []
    for value in status['obs']:
        for agasc_id in value['agasc_id']:
            row = value.copy()
            row['agasc_id'] = agasc_id
            rows.append(row)
    status['obs'] = rows
    return status


def parse_args(filename=None, bad_star_id=None, bad_star_source=None,
               obsid=None, status=None, comments='', agasc_id=None, mp_starcat_time=None,
               **_
               ):
    """
    Combine obs/bad-star status from file and from arguments.

    The arguments could be passed from and ArgumentParser doing::

        parse_args(vars(args))

    :param filename: str
    :param bad_star: int or list
    :param bad_star_source: int
    :param obs: int
    :param status: int
    :param comments: str
    :param agasc_id: int
    :param mp_starcat_time: str
    :return:
    """
    obs_status_override = []
    bad_star_id = list(np.atleast_1d(bad_star_id if bad_star_id else []))
    bad = []
    mags = []

    if bad_star_id and bad_star_source is None:
        raise RuntimeError('If you specify bad_star, you must specify bad_star_source')

    if filename is not None:
        status_file = _parse_obs_status_file(filename)
        bad = status_file['bad']
        obs_status_override = status_file['obs']
        mags = status_file['mags']

    if (obsid is not None or mp_starcat_time is not None) and status is not None:
        row = {
            'mp_starcat_time': mp_starcat_time,
            'agasc_id': agasc_id,
            'obsid': obsid,
            'status': status,
            'comments': comments
        }
        optional = ['obsid', 'mp_starcat_time', 'agasc_id', 'comments']
        obs_status_override.append(
            {key: row[key] for key in row if key not in optional or row[key] is not None}
        )

    for bs in bad_star_id:
        bad_dict = dict(bad)
        if bs in bad_dict and bad_dict[bs] != bad_star_source:
            raise RuntimeError('name collision: conflicting bad_star in file and in args')
        if (bs, bad_star_source) not in bad:
            bad.append((bs, bad_star_source))

    status = {'obs': obs_status_override, 'bad': bad, 'mags': mags}
    status = _sanitize_args(status)
    return status


def get_obs_status_parser():
    """
    Returns an argparse parser for the core options to update the 'bad' and 'obs' tables.

    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    status = parser.add_argument_group(
        'OBS/star status',
        'options to modify the "bads" and "obs" tables in AGASC supplement. '
        'Modifications to supplement happen before all magnitude estimates are made.'
    )
    status.add_argument('--obs-status-file',
                        help='YAML file with star/observation status. '
                             'More info at https://sot.github.io/agasc/supplement.html')
    status.add_argument('--mp-starcat-time',
                        help='Observation starcat time for status override. '
                             'Usually the mission planning catalog time')
    status.add_argument('--obsid', help='OBSID for status override.', type=int)
    status.add_argument('--agasc-id', help='AGASC ID for status override.', type=int)
    status.add_argument('--status', help='Status to override.')
    status.add_argument('--comments', help='Comments for status override.', default='')
    status.add_argument('--bad-star-id', help='AGASC ID of bad star.',
                        default=[], action='append', type=int)
    status.add_argument("--bad-star-source", type=int,
                        help="Source identifier indicating provenance.")

    return parser


def get_parser():
    """
    Returns the main parser for the update_obs_status script.

    :return: argparse.ArgumentParser
    """
    parse = argparse.ArgumentParser(
        description=__doc__,
        parents=[get_obs_status_parser()]
    )
    parse.add_argument("--output-dir",
                       default='.',
                       type=Path,
                       help="Directory containing agasc_supplement.h5 (default='.')")
    parse.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    parse.add_argument("--dry-run",
                       action="store_true",
                       help="Dry run (no actual file or database updates)")
    return parse


def update(args):
    """
    Update the 'bad', 'obs' and 'mags' tables as specified by the given arguments.

    The arguments are assumed to be from a parser like the one returned by `get_obs_status_parser`.

    Returns a list of AGASC IDs whose records were updated in the supplement
    """
    status = parse_args(
        filename=args.obs_status_file,
        **vars(args)
    )

    if status['mags']:
        update_mags_table(
            args.output_dir / 'agasc_supplement.h5',
            status['mags'],
            dry_run=args.dry_run
        )

    if status['obs']:
        update_obs_table(
            args.output_dir / 'agasc_supplement.h5',
            status['obs'],
            dry_run=args.dry_run
        )

    if status['bad']:
        add_bad_star(
            args.output_dir / 'agasc_supplement.h5',
            status['bad'],
            dry_run=args.dry_run
        )

    agasc_ids = sorted(set([o['agasc_id'] for o in status['obs']]
                           + [o[0] for o in status['bad']]))
    return agasc_ids


def main():
    """
    The main function for the update_obs_status script.
    """
    import kadi.commands
    kadi.commands.conf.commands_version = '1'

    args = get_parser().parse_args()

    status_to_int = {'ok': 0, 'good': 0, 'bad': 1}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    pyyaks.logger.get_logger(
        name='agasc.supplement',
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s"
    )

    cat.load()
    update(args)


if __name__ == '__main__':
    main()
