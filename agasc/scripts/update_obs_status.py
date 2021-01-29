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
import yaml
import numpy as np
from astropy import table
import pyyaks.logger

from agasc.supplement.magnitudes import star_obs_catalogs

logger = logging.getLogger('agasc.supplement')


def add_bad_star(bad_star_ids, bad_star_source, suppl_file, dry_run):
    """
    Update the 'bad' table of the AGASC supplement.

    :param bad_star_ids: int or list
    :param bad_star_source: int or list
    :param suppl_file: pathlib.Path
    :param dry_run: bool
    """
    if not bad_star_ids:
        logger.info('Nothing to update')
        return

    logger.info(f'updating "bad" table in {suppl_file}')

    bad_star_ids = np.atleast_1d(bad_star_ids).astype(int)

    try:
        dat = table.Table.read(str(suppl_file), format='hdf5', path='bad')
    except OSError as e:
        if str(e) == 'Path bad does not exist':
            logger.warning('creating "bad" table because it is missing')
            dat = table.Table(dtype=[('agasc_id', np.int32), ('source', np.int16)])
        else:
            raise

    if bad_star_source is None:
        bad_star_source = dat['source'].max() + 1
    else:
        bad_star_source = np.array(bad_star_source).astype(int)
    bad_star_ids, bad_star_source = np.broadcast_arrays(bad_star_ids, bad_star_source)

    update = False
    for agasc_id, source in zip(bad_star_ids, bad_star_source):
        if agasc_id not in dat['agasc_id']:
            dat.add_row((agasc_id, source))
            logger.info(f'Appending {agasc_id=} with {source=}')
            update = True
    if not update:
        return

    logger.info('')
    logger.info('IMPORTANT:')
    logger.info('Edit following if source ID is new:')
    logger.info('  https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually')
    logger.info('')
    logger.info('The wiki page also includes instructions for test, review, approval')
    logger.info('and installation.')
    if not dry_run:
        dat.write(str(suppl_file), format='hdf5', path='bad', append=True, overwrite=True)


def update_obs_table(filename, obs_status_override, dry_run=False):
    """
    Update the 'obs' table of the AGASC supplement.

    :param filename:
    :param obs_status_override: dict.
        Dictionary with OK flag for specific observations.
        Keys are (OBSID, AGASC ID) pairs, values are dictionaries like
        {'ok': True, 'comments': 'some comment'}
    :param dry_run: bool
    """
    if not Path(filename).exists():
        raise FileExistsError(f'AGASC supplement file does not exist: {filename}')

    if not obs_status_override:
        logger.info('Nothing to update')
        return

    try:
        obs_status = table.Table.read(filename, format='hdf5', path='obs')
    except OSError as e:
        if str(e) == 'Path obs does not exist':
            obs_status = {}
        else:
            raise

    obs_status = {
        (r['obsid'], r['agasc_id']): {'ok': r['ok'], 'comments': r['comments']}
        for r in obs_status
    }

    logger.info(f'updating "obs" table in {filename}')
    update = False
    for (obsid, agasc_id), status in obs_status_override.items():
        if (obsid, agasc_id) not in obs_status or obs_status[(obsid, agasc_id)] != status:
            logger.info(f'Appending {agasc_id=}, {obsid=}, {status=}')
            obs_status[(obsid, agasc_id)] = status
            update = True
    if not update:
        return

    if obs_status:
        t = list(zip(
            *[[oi, ai, np.uint(obs_status[(oi, ai)]['ok']), obs_status[(oi, ai)]['comments']]
              for oi, ai in obs_status]
        ))
        obs_status = table.Table(t, names=['obsid', 'agasc_id', 'ok', 'comments'])
    else:
        logger.info('creating empty obs table')
        dtype = [('obsid', int), ('agasc_id', int), ('ok', np.uint), ('comments', '<U80')]
        obs_status = table.Table(dtype=dtype)

    if not dry_run:
        obs_status.write(str(filename), format='hdf5', path='obs', append=True, overwrite=True)
    else:
        logger.info('dry run, not saving anything')


def _parse_obs_status_file(filename):
    """
    Parse a yaml file and return a dictionary.

    The dictionary will be of the form: {'obs': {}, 'bad': {}}

    :param filename:
    :return:
    """
    with open(filename) as fh:
        status = yaml.load(fh, Loader=yaml.SafeLoader)
    if 'obs' not in status:
        status['obs'] = []
    if 'bad' not in status:
        status['bad'] = {}
    for value in status['obs']:
        obs = value['obsid']
        if star_obs_catalogs.STARS_OBS is None:
            raise RuntimeError('Observation catalog is not initialized')

        if 'agasc_id' not in value:
            value['agasc_id'] = list(sorted(
                star_obs_catalogs.STARS_OBS[star_obs_catalogs.STARS_OBS['obsid'] == obs]['agasc_id']
            ))
        else:
            value['agasc_id'] = (list(np.atleast_1d(value['agasc_id'])))
        if 'comments' not in value:
            value['comments'] = ''

    status['obs'] = {
        (obs['obsid'], agasc_id): {
            'ok': obs['ok'],
            'comments': obs['comments']
        }
        for obs in status['obs'] for agasc_id in obs['agasc_id']
    }

    return status


def _parse_obs_status_args(filename=None, bad_star_id=None, bad_star_source=None,
                           obsid=None, status=None, comments='', agasc_id=None,
                           **_
                           ):
    """
    Combine obs/bad-star status from file and from arguments.

    The arguments could be passed from and ArgumentParser doing::

        parse_obs_status_args(vars(args))

    :param filename: str
    :param bad_star: int or list
    :param bad_star_source: int
    :param obs: int
    :param status: int
    :param comments: str
    :param agasc_id: int
    :return:
    """
    obs_status_override = {}
    bad_star_id = list(np.atleast_1d(bad_star_id if bad_star_id else []))
    bad = {}

    if bad_star_id and bad_star_source is None:
        raise RuntimeError('If you specify bad_star, you must specify bad_star_source')

    if filename is not None:
        status_file = _parse_obs_status_file(filename)
        bad = status_file['bad']
        obs_status_override = status_file['obs']

    if obsid is not None and status is not None:
        if agasc_id is None:
            from agasc.supplement.magnitudes import star_obs_catalogs
            agasc_ids = list(
                sorted(star_obs_catalogs.STARS_OBS[star_obs_catalogs.STARS_OBS['obsid'] == obsid][
                       'agasc_id'])
            )
        else:
            agasc_ids = sorted(list(np.atleast_1d(agasc_id)))

        for agasc_id in agasc_ids:
            obs_status_override[(obsid, agasc_id)] = {'ok': status, 'comments': comments}

    for bs in bad_star_id:
        if bs in bad and bad[bs] != bad_star_source:
            raise RuntimeError('name collision: conflicting bad_star in file and in args')
        bad[bs] = bad_star_source

    return {'obs': obs_status_override, 'bad': bad}


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
    status.add_argument('--obsid', help='Observation ID for status override.')
    status.add_argument('--agasc-id', help='AGASC ID for status override.')
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
    parse.add_argument("--data-root",
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
    Update the 'bad' and 'obs' tables as specified by the given arguments.

    The arguments are assumed to be from a parser like the one returned by `get_obs_status_parser`.
    """
    status = _parse_obs_status_args(
        filename=args.obs_status_file,
        **vars(args)
    )

    if status['obs']:
        update_obs_table(
            args.data_root / 'agasc_supplement.h5',
            status['obs'],
            dry_run=args.dry_run
        )

    if status['bad']:
        bad_star_ids, bad_star_source = zip(*status['bad'].items())
        add_bad_star(bad_star_ids,
                     bad_star_source,
                     args.data_root / 'agasc_supplement.h5',
                     dry_run=args.dry_run)
    return [o[1] for o in status['obs']]


def main():
    """
    The main function for the update_obs_status script.
    """
    args = get_parser().parse_args()

    status_to_int = {'true': 1, 'false': 0, 'ok': 1, 'good': 1, 'bad': 0}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    pyyaks.logger.get_logger(
        name='agasc.supplement',
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s"
    )

    star_obs_catalogs.load()
    update(args)


if __name__ == '__main__':
    main()
