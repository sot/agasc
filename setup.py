from distutils.core import setup

from agasc.version import version

setup(name='agasc',
      version=version,
      description='AGASC catalog access',
      author='Jean Connelly, Tom Aldcroft',
      author_email='taldcroft@head.cfa.harvard.edu',
      url='http://www.python.org/',
      packages=['agasc'],
      package_data={'agasc': ['ra_dec.npy']},
      )
