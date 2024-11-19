================================================
THE AXAF Guide and Acquisition Star Catalog V1.8
================================================

An all-sky astrometric and photometric catalog
prepared for the operation of Chandra
(formerly AXAF). version 1.8 July 2024

Introduction
============

The AXAF Guide and Acquisition Star Catalog (AGASC) is presented in
this set of FITS (Flexible Image Transport System) files, and may
be distributed in a variety of formats, including DAT (DDS format)
or CD-ROM (compact disc, read only memory in ISO 9660 format).  This
issue, Version 1.8, corresponds to the AGASC as completed
11 July 2024.

The primary source of data for the AGASC is the Guide Star Catalog
(GSC) for the Hubble Space Telescope (HST), version 1.1.  The release
number of AGASC will remain at 1 unless a different base catalog is
incorporated to reach to fainter magnitudes.  Historically, the
second catalog to be merged into AGASC1.1 was the Positions and Proper
Motions (PPM) Catalog.  The latter provides magnitudes, stellar types,
and proper motions for stars brighter than about V=12, but is far from
complete across the sky.  The third catalog merged into AGASC1.3 is
the Tycho Output Catalog (Tycho-1) from the Tycho star mapper on the
Hipparcos satellite.  Tycho-1 provided high precisions astrometric and
photometric data for 1,058,332 stars, nearly complete
to V=10.5.  The fourth catalog was the ACT Reference Catalog. This
subset of 988,758 TYC stars contained updated star positions and proper
motions for the epoch and equinox J2000.0, covering the entire sky.
The proper motions are from new reductions of the Astrographic
Catalogue combined with the positions in the Tycho Catalogue.
The 5th catalog merged in AGASC was the Tycho-2 catalog, with 2.5
million stars 90% complete to V=11.5mag, and including proper motions
from ACT and various other catalogs.  The major improvements
incorporated into AGASC1.5 include the following:

1. Repair of GSC1.1 star ID rollover problem in regions 3588, 5706, and 7899.
2. Merge of Tycho-2 position, mag, and color data for 2.5 million stars.
3. Recalibration of the estimated ACA magnitudes (MAG_ACA)
   and their errors (MAG_ACA_ERR) based on 1939 stars observed by the
   Chandra Aspect Camera Assembly (ACA) between 1999-Sep to 2000-Oct.
4. Update of remaining GSC1.1 positions by positions from
   the GSC-ACT catalog, reducing the systematic astrometric errors of
   faint GSC1.1 stars by about a factor of two.
5. Update of spoiler codes provides a more sophisticated
   measure of the effect of nearby stars on the best-fit centroid of a
   guide star.
6. Update of CLASS code to denote galaxies identified
   from pre-release J<12.5 2MASS galaxy catalog.

Version 1.6 was the third on orbit calibration of the AGASC. Using more
than 30000 aspect camera star observations from over the course of the
Chandra mission we were able to calculate the ACA response to a much
higher precision and over a broader band of star colors than previous
calibrations. AGASC 1.5 was deemed to require improvement for Chandra
flight operations after the discovery of a offset of up to 0.5 mags
between the predicted AGASC magnitude and the observed ACA magnitude
for red stars with a Tycho-2 color greater than 0.9.

Version 1.7 improves upon the 1.6 calibration (observed ACA magnitude
and its error) for red stars by including the i magnitude information
from the AAVSO (American Association of Variable Star Observers)
Photometric All-Sky Survey (APASS) DR9.

Version 1.8 includes improved star positions and proper motions from Gaia
DR3, as well as improved estimated magnitudes derived from Gaia DR3 photometry
and more than 94,000 ACA observed stars. 


Use of the AGASC 1.8 in Chandra Operations
==========================================

The Chandra Aspect Team uses an HDF5 version of the AGASC 1.8 data and
the Python agasc package https://sot.github.io/agasc/index.html. References
in this document to the agasc package refer to that Python code.
Contact aspect_help@cfa.harvard.edu if you would like to use the catalog
directly with that Python package.


Organization of the FITS Data Files
===================================

The top directory includes the files::

    ----------------------------------------------------------------
    agasc         - Directory for AGASC FITS region files.
    tables        - Directory for AGASC supporting tables.
    ----------------------------------------------------------------

In the tables subdirectory are these tables::

    ----------------------------------------------------
    regions       - Boundaries of GSC regions. FITS binary table
    lg_reg_x      - Index to large regions. FITS binary table
    sm_reg_x      - Index to small regions. FITS binary table
    neighbors     - Index of regions whose boundaries are within 1deg or
                    of the boundaries of each small region. ASCII
    boundaryfile  - Index of RA and DEC limits for each small region, in
                    decimal degrees. ASCII
    offset_lookup.fits
                    - FITS binary lookup table of the predicted ACA centroid
                    offsets (in arcsec) caused by a star of brightness
                    difference dm, and radial positional separation dr.
    outfilemap    - full path beginning with large region for each
                    small region. ASCII
    ----------------------------------------------------

As a result of merging of the GSC1.1 with the PPM and with the TYC,
and for expected merges with other catalogs in the future, we have
extended the AGASC format to include new information and references.
Published data include position, proper motion, epoch, parallax, a
magnitude, up to 2 colors, multiplicity and variability flags, and
source catalog IDs.  Cross-references are included separately for 1)
position 2) mag 3) color 4) proper motion (p.m.), 5) parallax, and 6)
variability. Derived data include MAG_ACA, MAG_ACA_ERR, a high
p.m. flag ASPQ2, and spoiler star codes ASPQ1, and ASPQQ1-6, which
potentially relieve the Star Selection Algorithm (SSA) of such
computations.

The AGASC stars are grouped with regions tables, as in the original
HST GSC.  The AGASC consists of about 9537 regions tables containing
about 2,000 objects each.  These will remain in FITS BINTABLE format,
with the directory structure described in FITS TABLE format.  Stars
from constituent catalogs that were not matched to the original GSC1.1
are included within the appropriate regions table based on region
boundaries in RA and DEC. Cross-references to the original star ID
numbers XREF_IDx are included from the original x=1-6 catalogs we
have matched.


Summary of the AGASC1.8 Format
==============================

Each FITS regions table in the AGASC1.8 consists of 3 parts, the
primary header, the table header, and the table data. The conventions
for FITS Binary Tables are detailed in Cotton, Tody and Pence (1995,
A&A, 113, 159), or at http://fits.nrao.edu/FITS.html

The length of the header information is the same for all the AGASC
regions tables.  That length is 5x2880= 14400bytes. After these 14400
bytes comes star data records.

The data for each star amounts to 122 bytes, in 47 data columns for
AGASC1.8.  Default values are -9999 or 0 where no data are available.
Many columns require data; these have no default values.  Another
exception is COLOR1, whose default value is 0.7000 for most stars,
or 1.5000 for stars with COLOR1 greater than 1.5000. Details on all
columns and their defaults below.

The FITS format data types and byte-lengths (8 bits to a byte) used
for each data item for each star are as follows::

  fmt    bytes fields  tot      type			   range
  ---------------------------------------------------------------
  A        1     0     0        character		-128 - 127
  B        1     8     8        unsigned integer	   0 - 255
  I        2     25    50       short integer	      -32768 - 32767
  J        4     6     24       long integer	 -2147483648 - 2147483647
  E        4     6     24       float variable	-9.22337e+18 - 9.22337e+18
  D        8     2     16       double variable	-1.70141e+38 - 1.70141e+38
  ----------------------------------------------------------------
                       122 bytes per star

Summary of the AGASC Version 1.8 Entries
----------------------------------------

Each of the FITS regions files in the AGASC1.8 will contain the
following fields for each entry::

    BYTES NAME - brief description

    4    AGASC_ID - a unique long integer used for identification.
        Currently a binary-packing of the region number, Hubble GSC star
        number, and Tycho Output Catalog identifier TYC3.
        No default value (must have an entry).

    8    RA - double variable expressing right ascension in decimal degrees.
        No default value (must have an entry).

    8    DEC - double variable expressing declination in decimal degrees.
        No default value (must have an entry).

    2    POS_ERR - short integer value of position uncertainty, in milli-arcsec.
        Default value of -9999 indicates no error available, or POS_ERR>32767.

    1    POS_CATID - unsigned integer identifying the source of the
        ra, dec, and pos_err.  Default value is 0.
            0 - no associated catalog
            1 - GSC1.1
            2 - PPM
            3 - Tycho Output Catalog (Tycho-1)
            4 - ACT
            5 - Tycho-2
            6 - GSC-ACT
            7 - Gaia DR3

    4    EPOCH - float variable identifying the epoch of the ra and dec
        measurements. Default value of -9999.0

    2    PM_RA - short integer variable expressing proper motion in ra in units of
        milli-arcsec per year.     Default value of -9999.

    2    PM_DEC - short integer variable expressing proper motion in dec in units
        of milli-arcsec per year.    Default value of -9999.

    1    PM_CATID - unsigned integer identifying the source of the
        pm_ra and pm_dec.  The codes are the same as listed for pos_catid.
        Default value is 0.

    2    PLX - short integer variable expressing parallax in units of
        milli-arcsec.    Default value of -9999.

    2    PLX_ERR - short integer variable expressing parallax error
        in units of milli-arcsec.    Default value of -9999.

    1    PLX_CATID - unsigned integer identifying the source of the
        pm_ra and pm_dec.  The codes are the same as listed for pos_catid.
        Default value is 0.

    4    MAG_ACA - float variable expressing the calculated magnitude in the AXAF
        ACA bandpass in units of magnitude. There is no default value.

    2    MAG_ACA_ERR - short integer expressing the uncertainty of mag_aca in
        units of 0.01mag. There is no default value.

    2    CLASS - short integer code identifying classification of entry.
        Default value of 0.
            0 - star
            1 - galaxy
            2 - blend or member of incorrectly resolved blend.
            3 - non-star
            5 - potential artifact
            6 - known multiple system
            7 - close to galaxy or other extended object
         >100 - bad star in AGASC supplement (only via agasc package query);
                class = 100 + bad star source ID.

        Note that code 1 is used only for a few hand-entered errata in
        or for galaxies with matches to preliminary 2MASS galaxy catalog.
        GSC1.1 galaxies   successfully processed by the STSci software have
        a classification of 3 (non-stellar).

    4    MAG - float variable expressing magnitude, in mags.  Spectral
        band for which magnitude is derived is summarized in entry MAG_BAND.
        There is no default value.

    2    MAG_ERR - short integer value of magnitude uncertainty, in
        0.01mag units. There is no default value.

    2    MAG_BAND - short integer code which identifies the spectral band
        for which the magnitude value is derived.
        There is no default value.

            Mag alpha Emulsion + Filter
            --- ----- ----------------
            0  0.72  IIIaJ + GG395
            1 -0.15  IIaD  + W12
            3  1.28  Tycho B
            4  0.106 Tycho V
            6 -0.10  IIaD  + GG495
            8 -0.71  103aE + Red Plexiglass
            10  0.78  yellow objective + IIaD + GG4
            11  1.16  blue objective +103aO
            12  1.16  blue objective +103aO
            13  0.13  yellow objective + 103aG + GG
            14  0.78  yellow objective + 103aG + GG
            16  0.00  IIIaJ + GG495
            18  0.72  IIIaJ + GG385
            21  0.00  PPM V mag
            22  1.00  PPM B mag
            23        Gaia DR3 G mag
            24        Gaia DR3 Rp mag
            25        Gaia DR3 Bp mag

    1    MAG_CATID - unsigned integer identifying the source of the
        mag, mag_err, and mag_band.  Codes are as follows:

            0 - no associated catalog
            1 - GSC1.1
            2 - PPM
            3 - Tycho Output Catalog (Tycho-1)
            4 - ACT
            5 - Tycho-2
            6 - GSC-ACT
            7 - Gaia DR3
          100 - Chandra ACA estimated magnitude (only via agasc package query)

    4    COLOR1 - float variable expressing the cataloged or estimated B-V color,
        used for mag_aca, in mag.  If no colors are available, the default
        value is 0.7000.  If the color is derived from Tycho-2 (C1_CATID=5) and
	that color is redder than (B-V)=1.5 then COLOR1 is set to 1.5000. This
	is the case for about 21,000 stars in AGASC 1.8. True cataloged color
	values are stored in COLOR2.

    2    COL0R1_ERR - short integer expressing the error in color1 in units of
        0.01 mag.  Default value of -9999.

    1    C1_CATID - unsigned integer identifying the source of color1 and
        color1_err.  The codes are the same as listed for pos_catid.
        Default value is 0.

    4    COLOR2 - float variable expressing a different color, in mag.
        For Tycho catalogs, this is the Tycho BT-VT color.
        Default value of -9999.0

    2    COLOR2_ERR - short integer expressing the error in color2, in
        units of 0.01mag.    Default value of -9999.

    1    C2_CATID - unsigned integer identifying the source of color2 and
        color2_err.  The codes are the same as listed for pos_catid.
        Default value is 0.

    4    RSV1 - APASS V - i magnitude (COLOR3). Default value of -9999.

    2    RSV2 - APASS V magnitude. Default value of -9999.

    1    RSV3 - unsigned integer indicating if the MAG_ACA and MAG_ACA_ERR
        were updated compared to AGASC1.6 (1 == updated, 0 == not updated).

    2    VAR - short integer code providing information on known or suspected
        variable stars.     Default value of -9999.
            1 - suspected variable, with a suspected amplitude variation < 2 mag
            2 - suspected variable, with a suspected amplitude variation > 2 mag
            3 - known variable, with an amplitude variation > 0.2 mag
            4 - known variable, with large amplitude ( > 2 mag), for which an
                ephemeris was necessary
            5 - known variable, with an amplitude variation < 0.2 mag

    1    VAR_CATID - unsigned integer code identifying the source of VAR
        Default value of 0.

    2    ASPQ1 - short integer spoiler code for aspect stars.
        An estimate, in 50milliarcsec units, of the worst centroid
        offset caused by any star within 80arcsec. The simulated PSF
        centroid offsets in the ACA are from offset_lookup.fits, indexed
        brightness difference dm, and radial positional separation dr.
        Default value of 0.

    2    ASPQ2 - short integer proper motion flag.
        Default value of 0.
            0 - unknown proper motion, or proper motion <500 milli-arcsec/year
            1 - proper motion >= 500 milli-arcsec/year

    2    ASPQ3 - short integer distance (for Tycho-2 stars only) to
        nearest Tycho-2 star, giving distance (in units of
        100milli-arcsec) computed for the epoch 1991.25.  The maximum
        value recorded for Tycho-2 stars is 999.
        Default value of 999.

    2    ACQQ1 - short integer indicating magnitude difference between the
        brightest star within 53.3" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    2    ACQQ2 - short integer indicating magnitude difference between the
        brightest star within 107" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    2    ACQQ3 - short integer indicating magnitude difference between the
        brightest star within 160.5" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    2    ACQQ4 - short integer indicating magnitude difference between the
        brightest star within 214" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    2    ACQQ5 - short integer indicating magnitude difference between the
        brightest star within 267.5" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    2    ACQQ6 - short integer indicating magnitude difference between the
        brightest star within 321" of this star, and this star, in units
        of 0.01 mags.     Default value of -9999.

    4    XREF_ID1 - long integer with the highest significant 32 bits of the Gaia DR3 ID.
        Default value of -1.

    4    XREF_ID2 - long integer which maps the entry to that in the PPM.
        Default value of -9999.

    4    XREF_ID3 - long integer which maps the entry to that in the Tycho Output
        Catalog (TYC2).  Default value of -9999.

    4    XREF_ID4 - long integer which maps the entry to that in the Tycho Output
        Catalog (TYC3).  Default value of -9999.

    4    XREF_ID5 - long integer with the lowest significant 32 bits of the Gaia DR3 ID.
        Default value of -9999.

    2    RSV4 - short integer which is the star number in the
        AGASC Version 1.0 (= GSC1.1).  This is not a unique identifier.
        Default value of -9999.

    2    RSV5 - short integer reserved for future use.  Default value of -9999.

    2    RSV6 - short integer reserved for future use.  Default value of -9999.

History of the AGASC Version 1.8
================================

The primary objective of the Chandra Aspect Camera Assembly (ACA) is to
measure the image positions of selected target stars and fiducial
lights in its field of view (FOV). The Chandra on board computer uses
gyro attitude data and ACA image centroids for real-time pointing.
Post-facto aspect determination is required for observations over 100
sec to compensate for the apparent motion of the X-ray image on the SI
focal plane.  When a maneuver is completed, at least 2 acquisition
stars must be acquired before acquiring guide stars and fiducial
lights.  Up to 8 images can be tracked, including the fid lights.  The
ground provides expected positions in the ACA FOV for these objects,
using the AGASC.  At least 5 stars brighter than m=10.2 in the ACA
instrumental mag (MAG_ACA) system should be provided from ground 95% of the
time, anywhere on the sky, for the predicted end of life (EOL) FOV of
1.79 square degrees.  To predict the ACA mag in advance, colors for
each star are required.  Proper motions (p.m.) are also advisable,
since high p.m. stars could move significantly over the extended
lifetime of the Chandra mission.   Parallax data are also advisable,
since parallaxes are not random, and in many cases will exceed
position errors.  Currently, the largest consistent published catalogs
providing colors and proper motions are the Positions and Proper
Motions (PPM) Catalog and the Tycho Output Catalog (TYC).

In 1996, we merged the PPM with the HST GSC1.1 to form AGASC1.1.
The optimal tolerance for positional matching of stars between the
GSC1.1 and the PPM was first determined, incorporating
p.m. information, and including all morphological classes (not just
stellar).  Studying a variety of celestial positions, including or
excluding non-stellar objects, we find an optimal positional matching
tolerance of r<=10arcsec.  To that separation, 295871 stars (99.74%)
are matched. To verify positionally matched stars, especially in more
crowded regions, we compare magnitudes between the GSC1.1 and PPM.
These mags are most often measured in different passbands.  For
simplicity, and greatest likelihood of compatibility with future
merged catalogs, we convert all magnitudes to the V band for
comparison using approximate B-V colors derived from the PPM spectral
types.  A magnitude tolerance of 2mag was allowed in the matching.
The large tolerance results from a variety of factors including at least
a) poor GSC1.1 magnitudes for bright stars, due to a poor mag-diameter
relation, halos and/or diffraction spikes b) large color uncertainties
since the PPM SpTypes may be crude and include no luminosity class, c)
random mag errors in either catalog.  To that tolerance,
295274 stars (99.54%) are matched. The differential histogram of
matched stars for all PPM stars with PPM visual mags turns over at
V=9, and is only a few hundred stars by V=12.

The original conversion from V to MAG_ACA was determined by convolving
the ACA bandpass with the Bruzual-Persson-Gunn-Stryker stellar
spectrophotometric atlas. This is an extension of the Gunn-Stryker
optical atlas (Gunn, J. E. & Stryker, L. L., 1983 ApJS, 52, 121) where
the spectral data have been extended into both the UV and the
infrared.  The IR data are from Strecker et al. (ApJ 41, 501, 1979)
and other unpublished sources.  Since the bandpass information for all
filters is normalized, the zeropoint for each filter was established
by convolution of the bandpass with a mag=0 spectrum of type G0V
(BD+26 3780 in the BPGS atlas, normalized to $V=0$). V and MAG_ACA mags
are then derived for each spectral type (SpType), resulting in a
V-MAG_ACA as a function of (B-V) color.  Newer calibrations
of ACA magnitude estimates for AGASC1.6 and AGASC1.7 are described below.

In 1997, we merged the TYC with AGASC1.1 to form AGASC1.2 This merging
was performed using the TYC ID codes TYC1 and TYC2, which are
cross-references to the HST GSC1.1 region number, and star number,
respectively.  Although all TYC stars appear to have GSC1.1
cross-references, there are 2 cases where stars are added to the
catalog.  First, where the TYC ID code TYC3 is greater than one, Tycho
has resolved into multiples an object previously unresolved in the
GSC1.1 Second, some TYC stars have no AGASC1.1 counterpart either from
the GSC1.1 or the PPM).  Again, data are checked to see if their
errors are smaller than those of data already in AGASC1.1 before being
substituted into AGASC1.2 However, PPM proper motions are assumed to
be superior due to their much longer baseline.  Positional data and
epoch are updated with the proper motion to Epoch 2000.  TYC parallax
measurements are all included for completeness, even though most are
not significant.  These will be used only for post-facto aspect
(image) reconstruction.  TYC V mags converted to Johnson V are
preferred, and TYC B mags are incorporated only in a few cases.
Johnson (B-V), as calculated in the TYC, are used for COLOR1 whenever
possible, with (BT-VT) now stored (redundantly) as COLOR2.
Multiplicity and variability information are also included.

Due to the short (less than 4 year) lifetime of the Hipparcos
mission, most of the proper motions included in the TYC are of low
significance.  Tycho positions make it the most accurate catalog of
comparable size at its epoch of observation, but its proper motions
degrade it to a sub-standard reference catalog in less than about 10
years.  The proper motions of the Tycho stars were improved from about
30 mas/year to about 3 mas/year by combining TYC positions
with AC2000 positions, yielding an average baseline of more than 80
years.  The ACT proper motion information and updated positions
were incorporated into AGASC1.3.

Actual ACA mags for stars of a wide range of spectral types were
accumulated during the first few months of the mission, and it
proved important to generate more accurate mags for the catalog from
the MAG and COLOR1 data. This recalibrated coefficients were derived
from a fit to observed Aspect Camera magnitudes for 271 stars observed
between 1999 Oct 03 - Nov 20, and AGASC1.3 was recalibrated to create
AGASC1.4.  Observed ACA mags for 1939 stars observed by the
Chandra Aspect Camera Assembly (ACA) between 1999-Sep to 2000-Oct
have been extracted and a calibrating polynomial refit by comparison
to colors from Tycho-1.

In AGASC1.6, MAG_ACA is derived from V and BT-VT (COLOR2) for all
stars with valid COLOR2 from Tycho-2.  MAG_ACA and MAG_ACA_ERR are
unchanged from AGASC 1.5 for stars not meeting these criteria.  A
seven node cubic spline was fit to the offset between the observed
magnitudes and V-Band magnitudes for 30238 ACA star observations.

In AGASC1.7, MAG_ACA is derived from V and BT-VT (COLOR2) and V-i
(COLOR3) for all stars with valid COLOR2 from Tycho-2 and valid COLOR3
from AAVSO (American Association of Variable Star Observers) Photometric
All-Sky Survey (APASS) DR9. MAG_ACA and MAG_ACA_ERR are unchanged from
AGASC1.6 for stars not meeting these criteria.

In early 2002, we merged in data from the Tycho-2 catalog, and
Tycho-2 supplement-1, described further below.  Tycho-2 data
supercedes data from all previously merged catalogs whenever it is
available.  Tycho-2 stars are matched to AGASC stars by the GSC star
ID, so that the final organization retains the original GSC star ID
and regions structure.

The average stellar surface density of unspoiled stars brighter than
MAG_ACA=10.2 with color information ((ASPQ1=0, CLASS=0, C1_CATID.ne.0)
is 9.5 stars per square degree in AGASC1.5

Near the galactic poles (b>80deg), where the stellar surface density
is lowest, there are 4.1 stars per square degree.  The desired figure
of merit (FOM) of 5.1 per square degree over 95% of the sky is thus
not quite achievable with these selection criteria from current
catalogs, and may not ever be (i.e. we are already nearly complete).
The current Chandra guide star selection includes stars without TYC
colors or PPM SpType information, which boosts the surface density,
but at this limiting ACA magnitude, such colors are available for 98%
of stars.


HST Guide Star Catalog
----------------------

The HST Guide Star Catalog (GSC), which has been constructed to support
the operational need of the Hubble Space Telescope for off-axis guide
stars, contains 18,819,291 objects in the seventh to sixteenth
magnitude range, of which more than 15 million are classified as
stars.

The GSC is primarily based on an all-sky, single epoch, single
passband collection of Schmidt plates.  For centers at +6 degrees and
north, a 1982 epoch "Quick V" survey was obtained by the Palomar
Observatory, while for southern fields, materials from the UK SERC J
survey (epoch approximately 1975) and its equatorial extension (epoch
approximately 1982) were used.

Photometry is available in the natural systems defined by the
individual plates in the GSC collection (generally J or V), and the
calibrations are done using B, V standards from the Guide Star
Photometric Catalog.  The overall quality of the photometry near the
standard stars is estimated from the fits and other tests to be 0.15
mag (one sigma, averaged over all plates), while the quality far from
the sequences is estimated from the all-sky plate-to-plate agreement
and from comparisons with independent photometric surveys to be about
0.30 mag (one sigma), with about 10% of the errors being greater than
0.50 mag.

Astrometry, at equinox J2000, is available at the epochs of the
individual plates used in the GSC; and the reductions to the reference
catalogs (AGK3, SAOC, or CPC, depending on the declination zone) use
third order expansions of the modeled plate and telescope effects.
Estimates of the overall external astrometric error, produced by
comparisons of independently measured positions without regard to
location on the GSC plates, are in the range 0.4 arc-sec to 0.6
arc-sec.

Further details concerning the HST GSC can be found in the following
publications:

1. The  Guide  Star  Catalog.  I.    Astronomical  and Algorithmic
   Foundations; Barry M. Lasker, Conrad R. Sturch, Brian J. McLean,
   Jane L. Russell, Helmut Jenkner, and Michael M. Shara;
   Astrophysical J. Suppl., 68, 1-90 (1988).

2. The  Guide  Star  Catalog.  II.   Photometric   and
   Astrometric Calibrations; Jane L. Russell, Barry M. Lasker,
   Brian J. McLean, Conrad R. Sturch, and Helmut Jenkner;
   Astronomical J., 99, 2059-2081 (1990).

3. The  Guide  Star  Catalog.  III.  Production, Database
   Organization,  and  Population  Statistics;   Helmut Jenkner,
   Barry M. Lasker, Conrad R. Sturch, Brian J. McLean, Michael
   M. Shara, and Jane L. Russell;  Astronomical, J., 99,
   2081-2154 (1990).

4. The table rev_1_1.tbl that accompanies the HST GSC1.1, as
   prepared by the Space Telescope Science Institute (ST ScI),
   3700 San Martin Drive,  Baltimore,  MD 21218,  USA.
   GSC  1.1  analysis and production were performed primarily by
   Jesse B.  Doggett, Daniel Egret, Brian J. McLean, and Conrad
   Sturch.



Positions and Proper Motions Catalog (PPM)
------------------------------------------

PPM North gives J2000 positions and proper motions of 181731 stars
north of -2.5 degrees declination.  The mean epoch is near 1931. The
average mean errors of the positions and proper motions are 0.27" and
0.43"/cen. On the average six measured positions are available per
star.  In addition to the positions and proper motions, the PPM
(North) contains the magnitude, the spectral type, the number of positions
included, the mean error of each component of the position and proper
motion, and the weighted mean epoch in each coordinate.

PPM South gives positions and proper motions of 197179 stars south of
about -2.5 degrees declination.  This net is designed to represent as
closely as possible the new IAU (1976) coordinate system on the sky,
as defined by the FK5 star catalogue (Fricke et al., 1988).


Further details concerning the PPM catalogs can be found in the following
publications:

1. Catalogue of Positions and Proper Motions; Roeser S., &
   Bastian U., 1988, Astron. Astrophys. Suppl. 74, 449

2. PPM South: A reference star catalogue for the southern
   hemisphere; Bastian, U., Roeser, S.,  Nesterov, V. V.,
   Polozhentsev, 	D. D., Potter, Kh. I., 1991, Astron.
   Astrophys. Suppl. 87, 159


TYCHO Output Catalog (TYC)
--------------------------

Colors are still needed for the majority of stars in AGASC1.1,
since merge with the PPM provided colors (from spectral types) for only
brightest 2% of the GSC1.1 to V=14.5.   Also, PPM-derived colors are
probably very uncertain, since they are interpolated from listed
Spectral Types with no reddening information. Merging the Tycho Output
Catalog with the existing AGASC1.1 provides reliable colors for
1,058,332 stars, nearly complete to V=10.5   In the current error
budget, using the highly accurate Tycho star positions should improve
the absolute aspect by 30-50% relative to GSC1.1 positions.

The Tycho Output Catalog from the Tycho star mapper on the Hipparcos
satellite, provides high precisions astrometric and photometric data
for 1,058,332 stars, nearly complete to V=10.5 Median astrometric
standard errors (in position, parallax, and annual proper motion) are
typically around 7mas for stars brighter than V_T_~9mag, and
approximately 25mas for V_T_~10.5mag, at the catalogue epoch
(J1991.25).  Astrometric errors for Tycho stars (typically 25
milliarcsec), are of the same order as parallaxes expected for many
bright (V<=8) stars typically selected by the Chandra Star Selection
Algorithm.  Since parallax is additive, it will dominate absolute
position errors unless incorporated in the Chandra image aspect solution.

Further details concerning the TYC catalog can be found in the following
publications:

1. The Tycho Reference Catalogue, Hog, E. et al. 1998,
   Astronomy and Astrophysics, 335, L65
2. The Tycho Catalogue, Hog, E. et al. 1997,
   Astronomy and Astrophysics, 323, L57

ACT Reference Catalog (ACT)
---------------------------

The ACT Reference Catalog contains 988,758 star positions and proper
motions covering the entire sky for the epoch and equinox J2000.0.
The proper motions are from new reductions of the Astrographic
Catalogue combined with the positions in the Tycho Catalogue.
The proper motions of the Tycho stars have thus been improved from about
30~mas/year to about 3~mas/year by recomputing them using the AC2000
data.  This combination of AC2000 and Tycho, called ACT Reference
Catalog, degrades much more slowly and is a valuable astronomical
dataset for applications potentially spanning decades.

The AC~2000 is a positional catalog recently compiled at The U.S. Naval
Observatory using the plate measures contained in the Astrographic
Catalogue (AC).   By the conclusion of the original AC project,
positions of 4.6 million stars had been measured, many as faint as
13th magnitude. These positions have an extremely early epoch; the
average epoch of an AC plate is 1907.  To compile the AC~2000, each of
the 22 zones making up the Astrographic Catalogue was reduced
independently using the Astrographic Catalog Reference Stars.

Further details concerning the ACT can be found in the following
publications:

1. The ACT Reference Catalog, Urban, S. E., Corbin, T. E.,
   and Wycoff, G. L. 1998, AJ, 115, 2161
2. The AC 2000: The Astrographic Catalogue on the System Defined by
   the Hipparcos Catalogue, Urban, S.E., et al. 1998, AJ, 115, 1212

TYCHO-2 Catalog
---------------

The Tycho-2 Catalogue is an astrometric reference catalogue containing
positions and proper motions as well as two-colour photometric data
for the 2.5 million brightest stars in the sky. Components of double
stars with separations down to 0.8 arcsec are included.

The Tycho-2 positions and magnitudes are based on precisely the same
observations as the Tycho-1 Catalogue (ESA SP-1200, 1997) collected by
the star mapper of the ESA Hipparcos satellite, but Tycho-2 is much bigger
and slightly more precise, owing to a more advanced reduction technique.

Proper motions precise to about 2.5 mas/yr are given as derived from
a comparison with the Astrographic Catalogue (AC) and 143 other
ground-based astrometric catalogues, all reduced to the Hipparcos
celestial coordinate system. For only about 100,000 stars, no proper
motion could be derived.

Tycho-2 supersedes Tycho-1, and the ACT and TRC catalogues based on Tycho-1.
The main Tycho-2 catalogue gives positions and at least one of B and V
for all stars at the epoch of observation. For most entries (96%)
proper motions (at epoch 2000) are also derived, using other
catalogues (mainly AC) and the corresponding mean positions at epoch
J2000.  When no proper motion has been derived, no mean position for
epoch J2000 is given.  Supplement-1 contains stars missing in Tycho-2,
but found in HIP or Tycho-1.  The supplement-1 only includes Tycho-1
stars of good quality and therefore the quality 9 (very poor) stars
and probable side-lobes were not included.

GSC-ACT Catalog
---------------

The original GSC1.1 positions have random position errors of about
0.4arcsec, but also systematic position errors of about 0.3arcsec,
due to errors in the reference catalogs (AGK3, SAO, CPC) and also
to their low stellar density. STScI performed a recalibration of the
GSC1.1, using the PPM catalog for a denser reference star network,
resulting in the GSC1.2: see
http://www-gsss.stsci.edu/gsc/gsc1/gsc12/DESCRIPTION.HTML.

In the GSC-ACT project, Bill Gray also recalibrated the GSC1.1,
but using the ACT (Astrographic Catalog/Tycho) data from the US Naval
Observatory. In the GSC-ACT,  GSC 1.1 systematic errors were reduced
via recalibration of 42 plate coefficients plate-by-plate, using the
proper-motion-corrected ACT stars for reference.
See http://www.projectpluto.com/gsc_act.htm

Here at the CXC, we matched (2arcsec search radius) both the GSC-ACT
and GSC1.2 against the catalog of ICRS defining source positions.
From 43 independent sources matched, the GSC1.2 showed a mean
positional difference of 0.40arcsec, RMS 0.35.  From 44 sources,
the GSC-ACT showed mean 0.28arcsec, RMS 0.25.  The denser reference
star network of the GSC-ACT results in a superior calibration,
this is the catalog we've chosen to improve star positions in the
AGASC that had only GSC1.1 (no Tycho or PPM) data.

2MASS Galaxy Catalog
--------------------

We use a pre-release catalog of extended objects from June 2001
kindly provided by Tom Jarrett (IPAC) and John Huchra (CfA).
The table used contains 41855 such objects, most of which
have not been verified as of this date by the human eye, or by match
to known galaxies.  In AGASC1.5, we required that any AGASC object that
has an extended 2MASS object within 5arcsec should have its CLASS
set to 1 (galaxy).  70% of 2MASS galaxies are thus matched.
CLASS=7 is used to denote objects within 3*r20 of 2MASS galaxies,
to mean that the object is close to a galaxy or other extended object.
The 2MASS project is a collaboration between The University of
Massachusetts and the Infrared Processing and Analysis Center (JPL/
Caltech). Funding is provided primarily by NASA and the NSF.
For information on 2MASS, see http://www.ipac.caltech.edu/2mass/

AAVSO Photometric All-Sky Survey (APASS)
----------------------------------------

We make use of data from the (American Association of Variable Star Observers (AAVSO)
Photometric All Sky Survey, DR9,
whose funding has been provided by the Robert Martin Ayers Sciences Fund
and by the NSF under grant AST-1412587. We matched the AGASC catalog
with the APASS catalog (2 arcsec search radius) and included a new color
information, V - i aka COLOR3 (APASS) in AGASC 1.7. COLOR2 (Tycho)
combined with COLOR3 (APASS) resulted in improved calibration of the
red spoiler stars.

Gaia DR3
--------

The Gaia DR3 catalog is the third data release from the European Space Agency's (ESA) Gaia mission.
The Gaia mission is designed to measure the positions, distances, and proper motions of stars with
unprecedented accuracy. We cross-matched the AGASC catalog with the Gaia DR3 catalog, and calibrated
the ACA magnitude estimates of more than 94,000 observed stars with the corresponding Gaia
magnitudes.

More details concerning Gaia can be found in the following resources:

   
1. Gaia Data Release 3, A. Avllenari et al. 2024, A&A, 674, A1
   https://doi.org/10.1051/0004-6361/202243940
2. https://www.cosmos.esa.int/web/gaia-users/archive/gdr3-documentation
3. https://gea.esac.esa.int/archive/


Acknowledgements
================

The AXAF Guide and Acquisition Star Catalog version 1.8 was prepared
from AGASC1.7 and Gaia DR3, primarily by Javier G Gonzalez,
Tom Aldcroft, and Jean Connelly. Thanks to the entire Star Selection and
Aspect Working Group for its input in the development and testing of this
catalog. The Chandra X-ray Center is supported through NASA Contract
NAS8-39073. Information about Chandra and the Chandra X-ray Observatory
Center may be found on the WWW at http://chandra.harvard.edu/
Detailed information about the catalog and its construction can be
obtained from the Chandra aspect web page at
https://cxc.harvard.edu/mta/ASPECT/agasc1p8 or by emailing:
aspect_help@cfa.harvard.edu
