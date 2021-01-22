#!/usr/bin/env python
"""
Update the 'bad' and 'obs' tables in the agasc supplement (agasc_supplement.h5).

This file is a supplement to the stable AGASC to inform star selection
and star catalog checking.

This script can add one or many bad stars to the bad star table, as well as observation status
information relevant for the magnitude estimation from acq data.
"""

from astropy import table
import yaml
import pathlib
import argparse
from agasc.scripts.add_bad_star import add_bad_star
import numpy as np
import logging
import pyyaks

from agasc.supplement.magnitudes import star_obs_catalogs

logger = logging.getLogger('agasc.supplement')


def parse_obs_status_file(filename):
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
        from agasc.supplement.magnitudes import star_obs_catalogs
        if star_obs_catalogs.STARS_OBS is None:
            raise RuntimeError('Observation catalog is not initialized')

        if 'agasc_id' not in value:
            value['agasc_id'] = list(
                sorted(star_obs_catalogs.STARS_OBS[star_obs_catalogs.STARS_OBS['obsid'] == obs]['agasc_id'])
            )
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


def parse_obs_status_args(filename=None, bad_star=[], bad_star_source=None,
                          obs=None, status=None, comments='', agasc_id=None,
                          **_
                          ):
    """
    Combine obs/bad-star status from file and from arguments.

    The arguments could be passed from and ArgumentParser doing::

        parse_obs_status_args(vars(args))

    :param filename: str
    :param bad_star: int
    :param bad_star_source: int
    :param obs: int
    :param status: int
    :param comments: str
    :param agasc_id: int
    :return:
    """
    obs_status_override = {}
    bad_star = (list(np.atleast_1d(bad_star)))
    bad = {}

    if bad_star and bad_star_source is None:
        raise RuntimeError('If you specify bad_star, you must specify bad_star_source')

    if filename is not None:
        status_file = parse_obs_status_file(filename)
        bad = status_file['bad']
        obs_status_override = status_file['obs']

    if obs is not None and status is not None:
        if agasc_id is None:
            from agasc.supplement.magnitudes import star_obs_catalogs
            agasc_ids = list(
                sorted(star_obs_catalogs.STARS_OBS[star_obs_catalogs.STARS_OBS['obsid'] == obs][
                       'agasc_id'])
            )
        else:
            agasc_ids = sorted(list(np.atleast_1d(agasc_id)))

        for agasc_id in agasc_ids:
            obs_status_override[(obs, agasc_id)] = {'ok': status, 'comments': comments}

    for bs in bad_star:
        if bs in bad and bad[bs] != bad_star_source:
            raise RuntimeError('name collision: conflicting bad_star in file and in args')
        bad[bs] = bad_star_source

    return {'obs': obs_status_override, 'bad': bad}


def update_obs_status(filename, obs_status_override, dry_run=False):
    """
    Update the magnitude table of the AGASC supplement.

    :param filename:
    :param obs_status_override: dict.
        Dictionary with OK flag for specific observations.
        Keys are (OBSID, AGASC ID) pairs, values are dictionaries like
        {'ok': True, 'comments': 'some comment'}
    :param dry_run: bool
    :return:
    """
    if not pathlib.Path(filename).exists():
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
        t = list(zip(*[[oi, ai, np.uint(obs_status[(oi, ai)]['ok']), obs_status[(oi, ai)]['comments']]
                       for oi, ai in obs_status]))
        obs_status = table.Table(t, names=['obsid', 'agasc_id', 'ok', 'comments'])
    else:
        logger.info('creating empty obs table')
        dtype = [('obsid', int), ('agasc_id', int), ('ok', np.uint), ('comments', '<U80')]
        obs_status = table.Table(dtype=dtype)

    if not dry_run:
        obs_status.write(str(filename), format='hdf5', path='obs', append=True, overwrite=True)
    else:
        logger.info('dry run, not saving anything')


def parser():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("--data-root",
                       default='.',
                       type=pathlib.Path,
                       help="Directory containing agasc_supplement.h5 (default='.')")
    parse.add_argument('--obs-status-override',
                       help='YAML file with star/observation status.'
                            'More info at https://sot.github.io/agasc/supplement.html')
    parse.add_argument('--obs', help='Observation ID for status override.')
    parse.add_argument('--agasc-id', help='AGASC ID for status override.')
    parse.add_argument('--status', help='Status to override.')
    parse.add_argument('--bad-star', help='Bad star.', default=[], action='append', type=int)
    parse.add_argument("--bad-star-source", type=int,
                       help=("Source identifier indicating provenance"
                             " (default=max existing source + 1)"))
    parse.add_argument('--comments', help='Comments for status override.', default='')
    parse.add_argument('--log-level',
                       default='info',
                       choices=['debug', 'info', 'warning', 'error'])
    parse.add_argument("--dry-run",
                       action="store_true",
                       help="Dry run (no actual file or database updates)")
    return parse


def main():
    the_parser = parser()
    args = the_parser.parse_args()

    status_to_int = {'true': 1, 'false': 0, 'ok': 1, 'good': 1, 'bad': 0}
    if args.status and args.status.lower() in status_to_int:
        args.status = status_to_int[args.status.lower()]

    pyyaks.logger.get_logger(
        name='agasc.supplement',
        level=args.log_level.upper(),
        format="%(asctime)s %(message)s"
    )

    star_obs_catalogs.load()

    status = parse_obs_status_args(
        filename=args.obs_status_override,
        **vars(args)
    )

    if status['obs']:
        update_obs_status(
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


if __name__ == '__main__':
    main()
