from distutils.core import setup

from agasc.version import version

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='agasc',
      version=version,
      description='AGASC catalog access',
      author='Jean Connelly, Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/agasc',
      packages=['agasc', 'agasc.tests'],
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
