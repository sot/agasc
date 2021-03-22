====================================
Maintaining the AGASC Supplement
====================================

Guidelines and Procedure to Update the Supplement
-------------------------------------------------

The AGASC supplement is updated weekly as part of the Aspect group operations.
The update and promotion process follows the steps described below.

Candidate supplement update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table of magnitudes in the supplement is updated by an automated cron task
that normally runs on Sunday afternoon each week. This process updates
magnitudes in the release candidate version of the supplement that resides in
``$SKA/data/agasc/rc``. It makes no changes to the flight version.

The processing creates a series of reports, including a report of "suspect"
observations, which require inspection and do not impact the supplement until
a disposition by the reviewer.

Reports are placed in
`<https://cxc.cfa.harvard.edu/mta/ASPECT/agasc/supplement_reports>`_.

Report review
^^^^^^^^^^^^^

The on-call ACA reviewer inspects the reports during the normal work week,
checking that the processing and indicated magnitude updates appear reasonable.
Most of the time this is the case and no further review is required.

For star observations that are suspect or are otherwise unusual, a disposition
will be made based on the report and by examining star image telemetry or other
available data as needed. Dispositions may include:

- Exclude or include specific star observations from the magnitude estimation.
  This informs the processing code about whether the data from a star
  observation can be used to compute the estimated star magnitude.
- Add the star to the AGASC supplement bad star table. This prevents it from
  being used in star selection and is independent of the magnitude estimate.
- Manually set a star magnitude to a given value in the case that the
  automated code is unable to determine a value. The most common case is where a
  guide star is never acquired despite repeated search attempts by OBC
  commanding. In this case the magnitude would typically be be set to the
  catalog ``MAXMAG`` to indicate a lower limit to the magnitude, and the star
  would normally be added to the bad stars table.

Other team members may be consulted at the discretion of the ACA reviewer.

Applying review
^^^^^^^^^^^^^^^

If no star dispositions were required then this step does not apply.

After deciding on the status of suspect star observations, the ACA reviewer
edits ``$SKA/data/agasc/rc/obs-status.yml`` accordingly. In order to apply the
changes the reviewer runs::

  agasc-supplement-tasks disposition

This command updates the release candidate version of the supplement in
``$SKA/data/agasc/rc`` as well as the weekly reports. The ACA reviewer then
checks the updated reports to validate that any dispositions have been applied
correctly.

Promotion
^^^^^^^^^

Once satisfied, the ACA reviewer runs the following to *schedule* the update
for promotion on the following Sunday::

  agasc-supplement-tasks schedule-promotion

This tool schedules the promotion by copying the relevant files into the
``$SKA/data/agasc/rc/promote`` directory. In the unlikely event of needing to
cancel the promotion, remove all files from ``$SKA/data/agasc/rc/promote``.

The actual promotion consists of copying all files from
``$SKA/data/agasc/rc/promote`` into the flight directory. This is done prior
to the weekly automated cron task to generate updated magnitude estimates.

Details
-------

The following sections provide more detailed information on the underlying
tools used to manage the AGASC supplement. In most cases these will not be
run manually during production processing.

Bad Stars and Star Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tables of bad stars and of star observation status in the AGASC supplement
can be updated using the `agasc-update-supplement`_ script.  This is normally
done using a YAML file, but the script also accepts command-line arguments to
specify the bad star and star observation information (more info below).

Calling the script with a YAML file can be done as follows::

    agasc-update-supplement --obs-status-file status.yml

An example `status.yml` file is:

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
    mags:
      - agasc_id: 1081092600
        mag_aca: 11.0
        mag_aca_err: 0.1

The above file will cause the script to add:

- AGASC IDs 77073552 and 23434 to the bad star list, with sources 9 and 10 respectively,
- AGASC ID 1081092600 to the "mags" table with a magnitude of 11.0 and uncertainty of 0.1,
- the observations of all stars observed in OBSID 56311 to the "obs" table with status=1,
- the observation of AGASC ID 806750112 in OBSID 56308 to the "obs" table with status=0,
- the observations of 1019348536 and 1019350904 in OBSID 11849 to the “obs” table,
  with status=1 and an optional comment string.

By default, the `agasc-update-supplement`_ script updates the supplement file in
the current working directory, but this can be specified in the command-line.

Alternatively, the following call adds a single bad star::

    agasc-update-supplement --bad-star-id 77073552 --bad-star-source 9

The following adds a single star observation::

    agasc-update-supplement --obs 11849 --agasc-id 1019348536 --status False

Updating via mica tools
"""""""""""""""""""""""

Prior to version 4.11.0 of the `agasc` package (including functionality to
generate, maintain, and use the AGASC supplement), the process for adding bad
stars to the supplement was spelled out in `the wiki page of the agasc module
Github repository
<https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually>`_.
The process in that page has been superceded and the page is now considered
archived.

Magnitude Supplement
^^^^^^^^^^^^^^^^^^^^

The AGASC magnitude supplement is automatically updated on a weekly basis using
the `agasc-update-magnitudes`_ script. A typical usage is as follows::

    agasc-update-magnitudes --report

That command does the following:

- Update/create the supplement file located in the current working directory (``agasc_supplement.h5``).
- Update/create a file with star-observation statistics (``mag_stats_obsid.fits``).
- Update/create a file with star statistics (``mag_stats_agasc.fits``).
- Produce HTML reports in the `supplement_reports/weekly` directory, relative to the ($CWD).

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Weekly reports are produced as magnitudes are estimated. Additionally, a report of `suspect` observations
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