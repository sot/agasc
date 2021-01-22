#!/usr/bin/env python
#
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Update the 'bad' table in agasc_supplement.h5.

This file is a supplement to the stable AGASC to inform star selection
and star catalog checking.

Currently this script only has the capability to add a bad star to the
bad star table.  It might end up including functionality to automatically
update another table with effective mags based on acq / guide history.

For process instructions see: https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually
"""

import os
import argparse
from pathlib import Path

import logging
import pyyaks.logger
from astropy.table import Table
import numpy as np

SKA = Path(os.environ['SKA'])

logger = logging.getLogger('agasc.supplement')


def parser():
    parse_ = argparse.ArgumentParser(description=__doc__)
    parse_.add_argument("--data-root",
                        default='.',
                        help=("Directory containing agasc_supplement.h5 (default='.')"))
    parse_.add_argument("--bad-star-id",
                        type=int,
                        help="AGASC ID of star to add to bad-star list")
    parse_.add_argument("--bad-star-source",
                        type=int,
                        help=("Source identifier indicating provenance (default=max "
                              "existing source + 1)"))
    parse_.add_argument("--log-level",
                        default=20,
                        help="Logging level (default=20 (info))")
    parse_.add_argument("--dry-run",
                        action="store_true",
                        help="Dry run (no actual file or database updates)")

    return parse_


def main(args=None):
    # Setup for updating the sync repository
    opt = parser().parse_args(args)

    # Set up logging
    loglevel = int(opt.log_level)
    logger = pyyaks.logger.get_logger(name='agasc.supplement', level=loglevel,
                                      format="%(message)s")

    data_root = Path(opt.data_root)
    suppl_file = data_root / 'agasc_supplement.h5'
    if suppl_file.exists():
        logger.info(f'Updating agasc_supplement at {suppl_file}')
    else:
        raise IOError(f'file {suppl_file.absolute()} not found')

    if opt.bad_star_id:
        add_bad_star(opt.bad_star_id, opt.bad_star_source, suppl_file, opt.dry_run)


def add_bad_star(bad_star_ids, bad_star_source, suppl_file, dry_run):
    if not bad_star_ids:
        logger.info('Nothing to update')
        return

    logger.info(f'updating "bad" table in {suppl_file}')

    bad_star_ids = np.atleast_1d(bad_star_ids).astype(int)
    dat = Table.read(str(suppl_file), format='hdf5', path='bad')

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


if __name__ == '__main__':
    main()
