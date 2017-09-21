# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

from agasc import __version__

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='agasc',
      version=__version__,
      description='AGASC catalog access',
      author='Jean Connelly, Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/agasc',
      packages=['agasc', 'agasc.tests'],
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
