"""
Perform tasks to update and promote the AGASC supplement.

The available tasks are:

* update-rc: copy files from $SKA/data/agasc/rc/promote into $SKA/data/agasc and
  update the supplement in $SKA/data/agasc/rc
* disposition: modify the observation status according to $SKA/data/agasc/rc/obs-status.yml
* schedule-promotion: schedule supplement promotion
"""

import argparse
import getpass
import os
import platform
import shutil
import subprocess
from email.mime.text import MIMEText
from pathlib import Path

from cxotime import CxoTime

from agasc.scripts import supplement_diff

AGASC_DATA = Path(os.environ["SKA"]) / "data" / "agasc"
SENDER = f"{getpass.getuser()}@{platform.uname()[1]}"


def email_promotion_report(filenames, destdir, to, sender=SENDER):
    date = CxoTime().date[:14]
    filenames = "- " + "\n- ".join([str(f) for f in filenames])

    msg = MIMEText(
        f"""
        The following files were promoted to {destdir} on {date}:\n{filenames}

        The corresponding changes are documented at
        https://cxc.cfa.harvard.edu/mta/ASPECT/agasc/supplement/agasc_supplement_diff.ecsv
        """
    )
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = "AGASC RC supplement promoted"
    p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
    p.communicate(msg.as_string().encode())


def update_rc():
    """
    Update the supplement in $SKA/data/agasc/rc
    """
    filenames = list((AGASC_DATA / "rc" / "promote").glob("*"))
    if (AGASC_DATA / "rc" / "promote" / "agasc_supplement.h5").exists():
        diff = supplement_diff.table_diff(
            AGASC_DATA / "agasc_supplement.h5",
            AGASC_DATA / "rc" / "promote" / "agasc_supplement.h5",
        )
        diff.write(
            "/proj/sot/ska/www/ASPECT/agasc/supplement/agasc_supplement_diff.ecsv",
            overwrite=True,
        )
    if filenames:
        for file in filenames:
            file.rename(AGASC_DATA / file.name)
        email_promotion_report(filenames, destdir=AGASC_DATA, to="aca@cfa.harvard.edu")

    subprocess.run(
        [
            "task_schedule3.pl",
            "-config",
            "agasc/task_schedule_update_supplement_rc.cfg",
        ],
        check=False,
    )


def disposition():
    """
    Apply obs-status dispositions from $SKA/data/agasc/rc/obs-status.yml.

    This actually schedules a task to run.
    """
    subprocess.run(
        [
            "task_schedule3.pl",
            "-config",
            "agasc/task_schedule_supplement_dispositions.cfg",
        ],
        check=False,
    )


def stage_promotion():
    """
    This function schedules files for promotion.

    It just copies the files into $SKA/data/agasc/rc/promote.
    The promotion task_schedule will move them to $SKA/data/agasc.
    """
    promote_dir = AGASC_DATA / "rc" / "promote"
    rc_dir = AGASC_DATA / "rc"
    promote_dir.mkdir(exist_ok=True)
    for filename in [
        "agasc_supplement.h5",
        "mag_stats_agasc.fits",
        "mag_stats_obsid.fits",
    ]:
        shutil.copy(rc_dir / filename, promote_dir / filename)


TASKS = {
    "update-rc": update_rc,
    "disposition": disposition,
    "schedule-promotion": stage_promotion,
}


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=TASKS)
    return parser


def main():
    args = get_parser().parse_args()
    TASKS[args.task]()


if __name__ == "__main__":
    main()
