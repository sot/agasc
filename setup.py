# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='agasc',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='AGASC catalog access',
      author='Jean Connelly, Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/agasc',
      packages=['agasc', 'agasc.tests'],
      package_data={'agasc.tests': ['data/*']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
