====================================
Updating the AGASC Supplement
====================================

Guidelines and Procedure to Update the Supplement
-------------------------------------------------

Currently, the guidelines for adding bad stars to the supplement are spelled out in
`the wiki page of the agasc module Github repository
<https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually>`_. They need to be
modified to agree with this documentation.

The supplement file is regularly updated as part of the Aspect group operations. The update and
promotion process follows these steps:

- Candidate supplement update. The table of magnitudes in the supplement is updated. Usually on a 
  Sunday. A series of reports are produced, including a report of "suspicious" observations, which
  require inspection. Reports are placed in
  `agasc/supplement_reports <https://cxc.cfa.harvard.edu/mta/ASPECT/agasc/supplement_reports>`_
- Report review. The aspect group inspects the reports as part of the weekly meeting. Upon reviewing
  the reports, a decisions is made on whether to:

  - exclude/include specific star observations from the magnitude estimation.
  - add a bad star
  - set a star magnitude to a given value

- Observation disposition. The aspect reviewer decides on the status of suspicious observations
  (and bad stars), edits $SKA/data/agasc/obs-status.yml accordingly,
  and runs `agasc-supplement-tasks disposition`.
- Approval.
- Promotion: The aspect reviewer runs `agasc-supplement-tasks promote`.


Bad Stars and Star Observations
-------------------------------

The tables of bad stars and of star observation status in the AGASC supplement are updated using the
`agasc-update-supplement`_ script. This script accepts command-line arguments to specify the bad star and
star observation information (more info below). It is also possible to use a yaml file. For example, calling the script
in this way::

    agasc-update-supplement --obs-status-file status.yml

With a `status.yml` file like the following:

.. code-block:: yaml

    obs:
      - obsid: 56311
        status: 1
      - obsid: 56308
        status: 0
        agasc_id: [806750112]
      - obsid: 11849
        status: 1
        agasc_id: [1019348536, 1019350904]
        comments: just removed them
    bad:
      77073552: 9
      23434: 10

will cause the script to add AGASC IDs 77073552 and 23434 to the bad star list, with sources 9 and 10 respectively.
The observations of all stars observed in OBSID 56311 will be added to the "obs" table with status=1, and the
observation of AGASC ID 806750112 in OBSID 56308 will be added with status=0. It is possible to specify a comment
string with the observation info.

By default, the `agasc-update-supplement`_ script updates the supplement file in the current working directory, but
this can be specified in the command-line.

Alternatively, the following call adds a single bad star::

    agasc-update-supplement --bad-star-id 77073552 --bad-star-source 9

and the following adds a single star observation::

    agasc-update-supplement --obs 11849 --agasc-id 1019348536 --status False

The Magnitude Supplement
------------------------

The AGASC magnitude supplement is automatically updated on a weekly basis using the `agasc-update-magnitudes`_ script.
A call like the following::

    agasc-update-magnitudes --report

Does the following:

- updates/creates the supplement file located in the current working directory (``agasc_supplement.h5``),
- updates/creates a file with star-observation statistics (``mag_stats_obsid.fits``),
- updates/creates a file with star statistics (``mag_stats_agasc.fits``),
- produces HTML reports in the `supplement_reports/weekly` directory, relative to the ($CWD)

Here are some other usage examples. Other useful command-line options are shown in the `agasc-update-magnitudes`_ section.
The following commands will update all observations since 2019:000 until the end of 2019, ignoring all observations
after the stopping time::

    agasc-update-magnitudes --start 2019:000 --stop 2020:000

By default, the script will update stars observed in the two weeks prior. Depending on OS, that is equivalent to one
of these::

    # CentOS 7
    agasc-update-magnitudes --start `date --date="14 days ago" "+%Y-%m-%dT%H:%M:%S"` --stop `date "+%Y-%m-%dT%H:%M:%S"`
    # OS-X
    agasc-update-magnitudes --start `date -v-2d "+%Y-%m-%dT%H:%M:%S"` --stop `date "+%Y-%m-%dT%H:%M:%S"`

This updates the magnitudes of all stars after a nominal start date (2003:000)::

    agasc-update-magnitudes --whole-history

This updates only the magnitudes of the AGASC IDs specified in the file agasc_ids.txt::

    agasc-update-magnitudes --agasc-id-file agasc_ids.txt

Magnitude Supplement Reports
----------------------------

Weekly reports are produced as magnitudes are estimated. Additionally, a report of `suspicious` observations
(over the last 90 days) is created in the `supplement_reports/suspect` directory, relative to the working directory,
by running::

    agasc-magnitudes-report

For this to work, the script needs to use two files that contain observed magnitude data.
These files are placed in the same directory as the supplement file whenever the supplement is updated.
The location of these files can also be specified in the command line. More information below.

Scripts
-------

.. _`agasc-update-supplement`:

:ref:`agasc-update-supplement`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.update_supplement.get_parser
   :prog: agasc-update-supplement


.. _`agasc-update-magnitudes`:

:ref:`agasc-update-magnitudes`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.update_mag_supplement.get_parser
   :prog: agasc-update-magnitudes


.. _`agasc-magnitudes-report`:

:ref:`agasc-magnitudes-report`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.mag_estimate_report.get_parser
   :prog: agasc-magnitudes-report

.. _`agasc-supplement-tasks`:

:ref:`agasc-supplement-tasks`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.supplement_tasks.get_parser
   :prog: agasc-supplement-tasks