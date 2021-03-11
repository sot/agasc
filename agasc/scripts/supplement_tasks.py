"""
Perform tasks to update and promote the AGASC supplement.

The available tasks are:

* update-rc: copy files from $SKA/data/agasc/rc/promote into $SKA/data/agasc and
  update the supplement in $SKA/data/agasc/rc
* disposition: modify the observation status according to $SKA/data/agasc/rc/obs-status.yml
* schedule-promotion: schedule supplement promotion
* promote: copy files from $SKA/data/agasc/rc/promote into $SKA/data/agasc
"""

import os
import subprocess
import argparse
from pathlib import Path
import shutil


AGASC_DATA = Path(os.environ['SKA']) / 'data' / 'agasc'


def update_rc():
    """
    Update the supplement in $SKA/data/agasc/rc
    """
    for file in (AGASC_DATA / 'rc' / 'promote').glob('*'):
        file.rename(AGASC_DATA / file.name)

    subprocess.run([
        'task_schedule3.pl',
        '-config',
        'agasc/task_schedule_update_supplement_rc.cfg'
    ])


def disposition():
    """
    Apply obs-status dispositions from $SKA/data/agasc/rc/obs-status.yml.

    This actually schedules a task to run.
    """
    subprocess.run([
        'task_schedule3.pl',
        '-config',
        'agasc/task_schedule_supplement_dispositions.cfg'
    ])


def stage_promotion():
    """
    This function schedules files for promotion.

    It just copies the files into $SKA/data/agasc/rc/promote.
    The promotion task_schedule will move them to $SKA/data/agasc.
    """
    promote_dir = AGASC_DATA / 'rc' / 'promote'
    rc_dir = AGASC_DATA / 'rc'
    if not promote_dir.exists():
        promote_dir.mkdir()
    for filename in ['agasc_supplement.h5', 'mag_stats_agasc.fits', 'mag_stats_obsid.fits']:
        if (rc_dir / filename).exists():
            shutil.copy(rc_dir / filename, promote_dir / filename)


TASKS = {
    'update-rc': update_rc,
    'disposition': disposition,
    'schedule-promotion': stage_promotion,
}


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('task', choices=TASKS)
    return parser


def main():
    args = get_parser().parse_args()
    TASKS[args.task]()


if __name__ == '__main__':
    main()
