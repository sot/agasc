# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


entry_points = {
    "console_scripts": [
        "agasc-magnitudes-report=agasc.scripts.mag_estimate_report:main",
        "agasc-update-magnitudes=agasc.scripts.update_mag_supplement:main",
        "agasc-update-supplement=agasc.scripts.update_supplement:main",
        "agasc-supplement-tasks=agasc.scripts.supplement_tasks:main",
        "agasc-supplement-diff=agasc.scripts.supplement_diff:main",
    ]
}


data_files = [
    (
        os.path.join("share", "agasc"),
        [
            "task_schedules/task_schedule_supplement_dispositions.cfg",
            "task_schedules/task_schedule_update_supplement_rc.cfg",
        ],
    )
]


setup(
    name="agasc",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description="AGASC catalog access",
    author="Jean Connelly, Tom Aldcroft",
    author_email="taldcroft@cfa.harvard.edu",
    url="http://cxc.harvard.edu/mta/ASPECT/tool_doc/agasc",
    packages=[
        "agasc",
        "agasc.supplement",
        "agasc.supplement.magnitudes",
        "agasc.tests",
        "agasc.scripts",
    ],
    package_data={
        "agasc.supplement.magnitudes": ["templates/*"],
        "agasc.tests": ["data/*"],
    },
    data_files=data_files,
    tests_require=["pytest"],
    cmdclass=cmdclass,
    entry_points=entry_points,
)
