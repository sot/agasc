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
      package_data=['agasc.tests': ['test_file_ra_0.0_dec_89.9_radius_0.6.fits',
                                    'test_file_ra_180.0_dec_-89.9_radius_0.6_1p6.fits',
                                    'test_file_ra_0.1_dec_0.0_radius_0.6_1p6.fits',
                                    'test_file_ra_180.0_dec_0.0_radius_0.6_1p6.fits',
                                    'test_file_ra_275.3647641740247_dec_8.099984164532414_radius_0.6_1p6.fits']],
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
