# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup
import os

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


entry_points = {
    'console_scripts': [
        'agasc-mag-estimate-report=agasc.scripts.mag_estimate_report:main',
        'agasc-supplement-update=agasc.scripts.update_mag_supplement:main',
        'agasc-supplement-bad-star=agasc.scripts.update_obs_status:main',
        'agasc-supplement-obs-status=agasc.scripts.update_obs_status:main',
    ]
}


data_files = [(os.path.join('share', 'agasc'), ['task_schedule.cfg'])]


setup(name='agasc',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='AGASC catalog access',
      author='Jean Connelly, Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/agasc',
      packages=['agasc', 'agasc.supplement', 'agasc.supplement.magnitudes',
                'agasc.tests', 'agasc.scripts'],
      package_data={'agasc.tests': ['data/*']},
      data_files=data_files,
      tests_require=['pytest'],
      cmdclass=cmdclass,
      entry_points=entry_points,
      )
