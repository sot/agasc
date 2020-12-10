====================================
Updating the AGASC Supplement
====================================

The AGASC magnitude supplement is updated on a weekly basis. A call like the following updates the supplement file
located in $SKA/data/agasc, and produces html reports in $SKA/www/ASPECT/agasc_supplement_reports::

    agasc-supplement-update --report

By default, this considers only the observations recorded in the two weeks prior. In other words,
this is equivalent to::

    agasc-supplement-update --start `date -I --date="14 days ago"` --stop `date -I`

The report of suspicious observations (over the last 90 days) is generated using::

    agasc-mag-estimate-report --start `date -I --date="90 days ago"` --stop `date -I`

In order to produce the reports, the script uses two files that contain observed magnitude data.
These files are placed in the same directory as the supplement file (usually $SKA/data/agasc).
The location of these files can be specified in the command line. More information below.

Scripts
-------

.. _`agasc-supplement-update`:

:ref:`agasc-supplement-update`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.update_mag_supplement.parser
   :prog: agasc-supplement-update


.. _`agasc-mag-estimate-report`:

:ref:`agasc-mag-estimate-report`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.mag_estimate_report.parser
   :prog: agasc-mag-estimate-report

Magnitude Supplement API
-------------------------

.. automodule:: agasc.supplement.magnitudes.mag_estimate
   :members:

.. automodule:: agasc.supplement.magnitudes.update_mag_supplement
   :members: