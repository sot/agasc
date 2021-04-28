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


Rubric for disposition of bad observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a list of annotations we have used when dispositioning bad observations.
When reviewing a suspect observation, the reviewer might look at telemetry to figure out whether
the star was ever acquired, it was in the field of view, and the magnitude was not significantly
affected by instrument effects. They serve as guide on how to disposition, and can refer to:

- Whether the star is seen in the field of view.
- Reasons for accepting/rejecting the observation.
- Confidence that the magnitude is correct or a limit.

Common annotations:

- **Faint star**. The star is discernible in image telemetry. E.g.: a faint star is poorly tracked
  even though it is always in the field of view and not spoiled/affected by imposters, in which case
  the magnitude is good.
- **Almost never acquired**. Star is discernible and residual < 5arcsec in a few sparse images. Maybe the
  magnitude information can still be used.
- **Never acquired**. The star was never discernible by eye in image telemetry. No magnitude
  information can be gleaned from telemetry. It is usually added as a bad star with source=9,
  and its magnitude set by hand.
- **Partially/slightly spoiled**. A spoiler shifted the centroid.
- **Out of view**. Something shifted the centroid so much that it was not within the window.
  Magnitude at these times is meaningless.
- **Tracking affected by imposter(s)**. The imposter drags the centroid away from the expected
  location, in which case the magnitude is close to correct when centroids match, and becomes larger
  as the angular distance increases (slot 7 in OBSID 50294, slot 1 in OBSID 48668).
- **Venus observation**.
- **Normal Sun Mode**. Nothing left after removing kadi.events.normal_suns time ranges.
- **Safe mode**. Nothing left after removing excluded kadi.events.safe_suns time ranges.
- **High Background Event**. Decided not to use the entire observation because of a large magnitude
  shift due to a high background event (we could include the event number).
- **High Background**. Even though there was no high background event, decided not to use the entire
  observation because of large magnitude shift due to what appears to be high background.
- **Marginally acceptable**. The magnitude estimate is not great, but will do, especially if this is
  the only observation of the star. Take this observation and be done with it.
- **Set mag to MAXMAG**. The magnitude is set by hand to MAXMAG from the starcheck catalog.
  One can navigate from the mag-stats dashboard to the mica OBSID page and from there to the
  starcheck catalog.
- **Set mag 12.0 +/- 0.1**. The magnitude is set by hand to some value determined by ACA reviewer.
- tracking interval is spurious. (What does this mean?)

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

This runs the update using the same arguments used for the last update/report::

    agasc-update-magnitudes --log-level debug --output-dir $SKA/data/agasc/rc --report \
        --args-file $SKA/data/agasc/rc/supplement_reports/weekly/latest/call_args.yml

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


Data products
-------------

There are a few data products resulting from the update of the supplement. The only essential one
is the supplement itself. All the others are not required for regular operations.

agasc_supplement.h5
^^^^^^^^^^^^^^^^^^^

The supplement is an HDF5 which contains the following tables:

bad
"""

====================  =======  ===========================
        name           dtype           description
====================  =======  ===========================
            agasc_id    int32         AGASC ID of the star
              source    int16  Bad star disposition source
====================  =======  ===========================


mags
""""

====================  =======  =================================================
        name           dtype                      description
====================  =======  =================================================
            agasc_id    int32                               AGASC ID of the star
             mag_aca  float32                 Star magnitude determined with ACA
         mag_aca_err  float32     Star magnitude uncertainty determined with ACA
       last_obs_time  float64  mp_starcat_time of the last observation of a star
====================  =======  =================================================

obs
"""

===============  =======  =========================================================================================
      name        dtype                                         description
===============  =======  =========================================================================================
mp_starcat_time      str     Timestamp from kadi.commands for starcat command preceding the dwell of an observation
       agasc_id    int32                                                                       AGASC ID of the star
          obsid    int32                             The OBSID corresponding to the dwell of an observation is made
         status    int32  Flag to tell include/excude the observation when estimating magnitude (0 means "include")
       comments      str                                                                           General comments
===============  =======  =========================================================================================


mag_stats_obsid.fits
^^^^^^^^^^^^^^^^^^^^

`mag_stats_obsid.fits` is an optional file that contains a single table. The table has one row for
each star-observation. This file is updated in each update, thus providing a summary of all
star-observations to date, but if it is created if it does not exist already.

====================  =======  ================================================================================================
        name           dtype                                             description
====================  =======  ================================================================================================
            agasc_id    int64                                                                              AGASC ID of the star
               obsid    int64                                     OBSID corresponding to the dwell when the observation is made
                slot    int64                                                                                       Slot number
                type   bytes3                                                                                       GUI/ACQ/BOT
     mp_starcat_time  bytes21  Unique timestamp (from kadi.commands) for starcat command preceding the dwell of an observation.
         timeline_id    int64                                           starcat command timeline_id from kadi.commands.get_cmds
              tstart  float64                                                          Dwell start time from kadi.events.manvrs
               tstop  float64                                                            Dwell end time from kadi.events.manvrs
      mag_correction  float64                                              Overall correction applied to the magnitude estimate
        responsivity  float64                                         Responsivity correction applied to the magnitude estimate
         droop_shift  float64                                          Droop shift correction applied to the magnitude estimate
             mag_aca  float32                                                         ACA star magnitude from the AGASC catalog
         mag_aca_err  float64                                             ACA star magnitude uncertainty from the AGASC catalog
                 row  float64          Expected row number, based on star location and yanf/zang from mica.archive.starcheck DB
                 col  float64          Expected col number, based on star location and yanf/zang from mica.archive.starcheck DB
             mag_img  float64                                             Magnitude estimate from image telemetry (uncorrected)
             mag_obs  float64                                                                      Estimated ACA star magnitude
         mag_obs_err  float64                                                          Estimated ACA star magnitude uncertainty
        aoacmag_mean  float64                                                                    Mean of AOACMAG from telemetry
         aoacmag_err  float64                                                      Standard deviation of AOACMAG from telemetry
         aoacmag_q25  float64                                                            1st quartile of AOACMAG from telemetry
      aoacmag_median  float64                                                                  Median of AOACMAG from telemetry
         aoacmag_q75  float64                                                            3rd quartile of AOACMAG from telemetry
          counts_img  float64                                 Raw counts from image telemetry, summed over the mouse-bit window
         counts_dark  float64                                 Expected counts from background, summed over the mouse-bit window
            f_kalman  float64              Fraction of all samples where AOACASEQ == "KALM" and AOPCADMD == "NPNT" (n_kalman/n)
             f_track  float64          Fraction of kalman samples with AOACIIR == "OK" and AOACFCT == "TRAK" (n_track/n_kalman)
               f_dr5  float64                Fraction of "track" samples with angle residual less than 5 arcsec (n_dr5/n_track)
               f_dr3  float64                Fraction of "track" samples with angle residual less than 3 arcsec (n_dr3/n_track)
                f_ok  float64                              Fraction of all samples with (kalman & track & dr5) == True (n_ok/n)
                 q25  float64                                                               1st quartile of estimated magnitude
              median  float64                                                                     Median of estimated magnitude
                 q75  float64                                                               1st quartile of estimated magnitude
                mean  float64                                                                       Mean of estimated magnitude
            mean_err  float64                                                     Uncrtainty in the mean of estimated magnitude
                 std  float64                                                         Standard deviation of estimated magnitude
                skew  float64                                                                   Skewness of estimated magnitude
                kurt  float64                                                                   Kurtosis of estimated magnitude
              t_mean  float64                                               Mean of estimated magnitude after removing outliers
          t_mean_err  float64                            Uncertainty in the mean of estimated magnitude after removing outliers
               t_std  float64                                 Standard deviation of estimated magnitude after removing outliers
              t_skew  float64                                           Skewness of estimated magnitude after removing outliers
              t_kurt  float64                                           Kurtosis of estimated magnitude after removing outliers
                   n    int64                                                                                 Number of samples
                n_ok    int64                                             Number of samples with (kalman & track & dr5) == True
            outliers    int64                                                                     Number of outliers (+- 3 IQR)
 lf_variability_100s  float64                                            Rolling mean of OK magnitudes with a 100 second window
 lf_variability_500s  float64                                            Rolling mean of OK magnitudes with a 500 second window
lf_variability_1000s  float64                                           Rolling mean of OK magnitudes with a 1000 second window
             tempccd  float64                                                                                   CCD temperature
             dr_star  float64                                                                                    Angle residual
              obs_ok     bool                                                 Boolean flag: everything OK with this observation
         obs_suspect     bool                                                       Boolean flag: this observation is "suspect"
            obs_fail     bool                   Boolean flag: a processing error when estimating magnitude for this observation
            comments  bytes54                                                                                  General comments
                   w  float64                                                      Weight to be used on a weighted mean (1/std)
      mean_corrected  float64                                    Corrected mean used in weighted mean (t_mean + mag_correction)
       weighted_mean  float64                                         Mean weighted by inverse of standard deviation (mean/std)
====================  =======  ================================================================================================


mag_stats_agasc.fits
^^^^^^^^^^^^^^^^^^^^

`mag_stats_agasc.fits` is an optional file that contains a single table. The table has one row for
each star. This file is updated in each update, thus providing a summary of all
star-observations to date, but if it is created if it does not exist already.

====================  =======  ======================================================================================================================================
        name           dtype                                                                description
====================  =======  ======================================================================================================================================
       last_obs_time  float64                                                                      CXC seconds corresponding to the last mp_starcat_time for the star
            agasc_id    int64                                                                                                                    AGASC ID of the star
             mag_aca  float64                                                                                               ACA star magnitude from the AGASC catalog
         mag_aca_err  float64                                                                                   ACA star magnitude uncertainty from the AGASC catalog
             mag_obs  float64                                                                                                            Estimated ACA star magnitude
         mag_obs_err  float64                                                                                                Estimated ACA star magnitude uncertainty
         mag_obs_std  float64                                                                                         Estimated ACA star magnitude standard deviation
               color  float64                                                                                                       Star color from the AGASC catalog
            n_obsids    int64                                                                                                     Number of observations for the star
       n_obsids_fail    int64                                                                                   Number of observations which give an unexpected error
    n_obsids_suspect    int64                                                           Number of observations deemed "suspect" and ignored in the magnitude estimate
         n_obsids_ok    int64                                                                             Number of observations considered in the magnitude estimate
          n_no_track    int64                                                                                 Number of observations where the star was never tracked
                   n    int64                                                                                              Total number of image samples for the star
                n_ok    int64                                                               Total number of image samples included in magnitude estimate for the star
                f_ok  float64                                                                            Fraction of the total samples included in magnitude estimate
              median  float64                                                                                                  Median magnitude over OK image samples
         sigma_minus  float64                                                                                       15.8% quantile of magnitude over OK image samples
          sigma_plus  float64                                                                                       84.2% quantile of magnitude over OK image samples
                mean  float64                                                                                                 Mean of magnitude over OK image samples
                 std  float64                                                                                   Standard deviation of magnitude over OK image samples
   mag_weighted_mean  float64                                               Average of magnitudes over observations, weighed by the inverse of its standard deviation
    mag_weighted_std  float64                                                                                              Uncertainty in the weighted magnitude mean
              t_mean  float64                                                                       Mean magnitude after removing outliers on a per-observation basis
               t_std  float64                                                         Magnitude standard deviation after removing outliers on a per-observation basis
           n_outlier    int64                                                                                  Number of outliers, removed on a per-observation basis
            t_mean_1  float64                                                                                          Mean magnitude after removing 1.5*IQR outliers
             t_std_1  float64                                                                            Magnitude standard deviation after removing 1.5*IQR outliers
         n_outlier_1    int64                                                                                                              Number of 1.5*IQR outliers
            t_mean_2  float64                                                                                            Mean magnitude after removing 3*IQR outliers
             t_std_2  float64                                                                              Magnitude standard deviation after removing 3*IQR outliers
         n_outlier_2    int64                                                                                                                Number of 3*IQR outliers
       selected_atol     bool                                                                                                            abs(mag_obs - mag_aca) > 0.3
       selected_rtol     bool                                                                                                abs(mag_obs - mag_aca) > 3 * mag_aca_err
selected_mag_aca_err     bool                                                                                                                       mag_aca_err > 0.2
      selected_color     bool                                                                                                         (color == 1.5) | (color == 0.7)
          t_mean_dr3  float64                Truncated mean magnitude after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
           t_std_dr3  float64  Truncated magnitude standard deviation after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
      t_mean_dr3_not  float64
       t_std_dr3_not  float64
            mean_dr3  float64                          Mean magnitude after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
             std_dr3  float64            Magnitude standard deviation after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
               f_dr3  float64                                                                   Fraction of OK image samples with angular residual less than 3 arcsec
               n_dr3    int64                                                                     Number of OK image samples with angular residual less than 3 arcsec
      n_dr3_outliers    int64            Number of magnitude outliers after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
          median_dr3  float64                        Median magnitude after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
     sigma_minus_dr3  float64             15.8% quantile of magnitude after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
      sigma_plus_dr3  float64             84.2% quantile of magnitude after removing outliers and samples with angular residual > 3 arcsec on a per-observation basis
          t_mean_dr5  float64                Truncated mean magnitude after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
           t_std_dr5  float64  Truncated magnitude standard deviation after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
      t_mean_dr5_not  float64
       t_std_dr5_not  float64
            mean_dr5  float64                          Mean magnitude after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
             std_dr5  float64            Magnitude standard deviation after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
               f_dr5  float64                                                                   Fraction of OK image samples with angular residual less than 5 arcsec
               n_dr5    int64                                                                     Number of OK image samples with angular residual less than 5 arcsec
      n_dr5_outliers    int64            Number of magnitude outliers after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
          median_dr5  float64                        Median magnitude after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
     sigma_minus_dr5  float64             15.8% quantile of magnitude after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
      sigma_plus_dr5  float64             84.2% quantile of magnitude after removing outliers and samples with angular residual > 5 arcsec on a per-observation basis
====================  =======  ======================================================================================================================================
