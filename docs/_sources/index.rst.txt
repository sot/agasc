.. Agasc documentation master file, created by
   sphinx-quickstart on Thu May 16 17:03:14 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AGASC Catalog Description and History
-------------------------------------
.. toctree::
   :maxdepth: 2

   catalog-description


Getting started
---------------

The most common usage of the ``agasc`` package is getting the AGASC catalog data
for stars near a location on the sky or by AGASC ID.

The three main functions are:

- :func:`~agasc.agasc.get_agasc_cone`: Get stars within a radius of RA, Dec
- :func:`~agasc.agasc.get_star`: Get information for one star by AGASC ID
- :func:`~agasc.agasc.get_stars`: Get information for a list of stars by AGASC ID

``agasc`` package API documentation
-----------------------------------
.. toctree::

   api

Topics For Maintainers
----------------------

.. toctree::
   :maxdepth: 2

   supplement
   maintainer_api