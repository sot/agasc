====================================
Updating the AGASC Supplement
====================================

Guidelines and Procedure to Update the Supplement
-------------------------------------------------

Currently, the guidelines for adding bad stars to the supplement are spelled out in
`the wiki page of the agasc module Github repository
<https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually>`_.

Additionally, the table of magnitudes in the supplement is updated regularly. Whenever the magnitudes are updated, a
report is placed in `agasc/supplement_reports <https://cxc.cfa.harvard.edu/mta/ASPECT/agasc/supplement_reports>`_.
Upon reviewing these reports we can decide to exclude/include specific star observations from the magnitude estimation.
To accomplish this, we need to add the corresponding observations to the "obs" table in the supplement.
The approval procedure for this update is not yet specified, but after approval the supplement will be updated as
detailed below.

Bad Stars and Star Observations
-------------------------------

The tables of bad stars and of star observation status in the AGASC supplement are updated using the
`agasc-supplement-obs-status`_ script. This script accepts command-line arguments to specify the bad star and
star observation information (more info below). It is also possible to use a yaml file. For example, calling the script
in this way::

    agasc-supplement-obs-status --obs-status-override status.yml

With a `status.yml` file like the following:

.. code-block:: yaml

    obs:
      - obsid: 56311
        ok: false
      - obsid: 56308
        ok: true
        agasc_id: [806750112]
      - obsid: 11849
        ok: false
        agasc_id: [1019348536, 1019350904]
        comments: just removed them
    bad:
      77073552: 9
      23434: 10

will cause the script to add AGASC IDs 77073552 and 23434 to the bad star list, with sources 9 and 10 respectively.
The observations of all stars observed in OBSID 56311 will be added to the "obs" table with status=False, and the
observation of AGASC ID 806750112 in OBSID 56308 will be added with status=True. It is possible to specify a comment
string with the observation info.

By default, the `agasc-supplement-obs-status`_ script updates the supplement file in the current working directory, but
this can be specified in the command-line.

Alternatively, the following call adds a single bad star::

    agasc-supplement-obs-status --bad-star 77073552 --bad-star-source 9

and the following adds a single star observation::

    agasc-supplement-obs-status --obs 11849 --agasc-id 1019348536 --status False

The Magnitude Supplement
------------------------

The AGASC magnitude supplement is automatically updated on a weekly basis using the `agasc-supplement-update`_ script.
Here are some usage examples. Other useful command-line options are shown in the `agasc-supplement-update`_ section.

A call like the following updates the supplement file located in the current working directory, and produces html
reports in the `supplement_reports/weekly` directory, relative to the working directory::

    agasc-supplement-update --report

The following commands will update all observations since 2019:000 until the end of 2019, ignoring all observations
after the stopping time::

    agasc-supplement-update --start 2019:000 --stop 2020:000

By default, the script will update stars observed in the two weeks prior. Depending on OS, that is equivalent to one
of these::

    # CentOS 7
    agasc-supplement-update --start `date --date="14 days ago" "+%Y-%m-%dT%H:%M:%S"` --stop `date "+%Y-%m-%dT%H:%M:%S"`
    # OS-X
    agasc-supplement-update --start `date -v-2d "+%Y-%m-%dT%H:%M:%S"` --stop `date "+%Y-%m-%dT%H:%M:%S"`

This updates the magnitudes of all stars after a nominal start date (2003:000)::

    agasc-supplement-update --whole-history

This updates only the magnitudes of the AGASC IDs specified in the file agasc_ids.txt::

    agasc-supplement-update --agasc-id-file agasc_ids.txt

Magnitude Supplement Reports
----------------------------

Weekly reports are produced as magnitudes are estimated. Additionally, a report of `suspicious` observations
(over the last 90 days) is created in the `supplement_reports/suspect` directory, relative to the working directory,
by running::

    agasc-mag-estimate-report

For this to work, the script needs to use two files that contain observed magnitude data.
These files are placed in the same directory as the supplement file whenever the supplement is updated.
The location of these files can also be specified in the command line. More information below.

Scripts
-------

.. _`agasc-supplement-obs-status`:

:ref:`agasc-supplement-obs-status`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.update_obs_status.parser
   :prog: agasc-supplement-obs-status


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

.. _`agasc-supplement-bad-star`:

:ref:`agasc-supplement-bad-star`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: agasc.scripts.add_bad_star.parser
   :prog: agasc-supplement-bad-star
