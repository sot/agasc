.. Agasc documentation master file, created by
   sphinx-quickstart on Thu May 16 17:03:14 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AGASC Catalog Description and History
-------------------------------------
.. toctree::
   :maxdepth: 2

   catalog-description

- :doc:`catalog-description-1p7`

.. toctree::
   :hidden:

   catalog-description-1p7


Getting started
---------------

The most common usage of the ``agasc`` package is getting the AGASC catalog data
for stars near a location on the sky or by AGASC ID.

The three main functions are:

- :func:`~agasc.agasc.get_agasc_cone`: Get stars within a radius of RA, Dec
- :func:`~agasc.agasc.get_star`: Get information for one star by AGASC ID
- :func:`~agasc.agasc.get_stars`: Get information for a list of stars by AGASC ID

AGASC catalog order
~~~~~~~~~~~~~~~~~~~

Catalogs with version 1.7 or before are ordered by increasing Declination. Cone search
is done by selecting a band of Declination and then searching for stars within the
required spatial cone.

Starting with version 1.8, catalogs are ordered by `HEALpix
<https://healpix.sourceforge.io/>`_ index. HEALpix is a library that provides a fast way
to uniquely tile a sphere with tiles of uniform area, providing a map between any point
on the sphere and the index number. The HEALpix index is a pixel number between 0 and
12*NSIDE**2-1, where NSIDE is the HEALpix resolution parameter. For AGASC 1.8, NSIDE=64.

HEALpix-ordered catalogs include a ``healpix_index`` table that provides the row range
for each HEALpix index. Cone search is done by finding the HEALpix pixels that overlap
the required spatial cone and extracting the stars from the catalog that are in those
pixels. This strategy is faster than the Declination-ordered catalog search and requires
less memory.

Note that for testing, some version 1.7 catalogs are available in HEALpix order. These
are denoted with a ``_healpix`` suffix in the filename.

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
