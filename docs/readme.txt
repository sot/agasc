

            THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.7
                                                                
             An all-sky astrometric and photometric catalog     
                 prepared for the operation of Chandra 
                 (formerly AXAF). version 1.7 September 2018

                                                                      
    These files constitute the AXAF Guide and Acquisition Star Catalog
    - Version 1.7, with an issue date of 18 September 2018

    The AXAF Guide and Acquisition Star Catalog (AGASC) was prepared
    by the AXAF Science Center (ASC), Smithsonian Astrophysical
    Observatory, 60 Garden St., Cambridge, MA 02138.              

    AGASC1.7 may be provided in a variety of formats, including
    DAT tape CD-ROMs in the ISO 9660 format.


CONTENTS:
        1. INTRODUCTION
        2. ORGANIZATION OF THE DATA FILES
        3. FORMAT
        4. ACKNOWLEDGEMENTS

                                                                      
                        1. INTRODUCTION                        


The AXAF Guide and Acquisition Star Catalog (AGASC) is presented in
this set of FITS (Flexible Image Transport System) files, and may
be distributed in a variety of formats, including DAT (DDS format)
or CD-ROM (compact disc, read only memory in ISO 9660 format).  This
issue, Version 1.7, corresponds to the AGASC as completed 18 September 2018.

AGASC1.7 can be roughly considered as the superset of the best data
available for each of 18865968 listed stars, taken from the following
catalogs: Tycho-2 and GSC-ACT.  Additional derived quantities are
included that are useful for Chandra operations, primarily the
expected magnitude in the Chandra Aspect Camera (Ball Aerospace &
Technologies Corp).  Further details of AGASC1.7 history and
construction are elaborated in the comments.txt file of the table
subdirectory.  

                                                                     

                      2.   ORGANIZATION OF THE DATA FILES          

                                        
The top directory includes the files
----------------------------------------------------------------        
  readme.txt    - Introduction. ASCII
  agasc         - Directory for AGASC FITS region files.                      
  tables        - Directory for AGASC supporting tables.                    
----------------------------------------------------------------

 In the tables subdirectory are these tables:
----------------------------------------------------         
  comments.txt  - Introduction and general comments. ASCII  
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
extended the GSC format to include new information and references.
Published data include position, proper motion, epoch, parallax, a
magnitude, up to 2 colors, multiplicity and variability flags, and
source catalog IDs.  Cross-references are included separately for 1)
position 2) mag 3) color 4) proper motion (p.m.), 5) parallax, and 6)
variability. Derived data include MAG_ACA, MAG_ACA_ERR, a high
p.m. flag ASPQ2, and spoiler star codes ASPQ1, and ASPQQ1-6, which
potentially relieve the Chandra Off-Line System Star Selection
Algorithm (SSA) of such computations.  

 The AGASC stars are grouped with regions tables, as in the original
HST GSC.  The AGASC consists of about 9537 regions tables containing
about 2,000 objects each.  These will remain in FITS BINTABLE format,
with the directory structure described in FITS TABLE format.  Stars
from constituent catalogs that were not matched to the original GSC1.1
are included within the appropriate regions table based on region
boundaries in RA and DEC. Cross-references to the original star ID
numbers XREF_IDx are included from the original x=1-6 catalogs we 
have matched.


		3. SUMMARY OF THE AGASC1.7 FORMAT

  Each FITS regions table in the AGASC1.7 consists of 3 parts, the
primary header, the table header, and the table data. The conventions
for FITS Binary Tables are detailed in Cotton, Tody and Pence (1995,
A&A, 113, 159), or at http://fits.nrao.edu/FITS.html

The length of the header information is the same for all the AGASC
regions tables.  That length is 5x2880= 14400bytes. After these 14400
bytes comes star data records. 

 The data for each star amounts to 122 bytes, in 47 data columns for
AGASC1.7.  Default values are -9999 or 0 where no data are available.
Many columns require data; these have no default values.  Another 
exception is COLOR1, whose default value is 0.7000 for most stars,
or 1.5000 if COLOR1 for stars redder than 1.5000.  Details on all
columns and their defaults below.


 The FITS format data types and byte-lengths (8 bits to a byte) used
for each data item for each star are as follows:
  

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



SUMMARY OF THE AGASC version 1.7 ENTRIES

 Each of the FITS regions files in the AGASC1.7 will contain the
 following fields for each entry:


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

  Note that code 1 is used only for a few hand-entered errata; galaxies
  successfully processed by the STSci software have a classification of 3
  (non-stellar).
  
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
 
1    MAG_CATID - unsigned integer identifying the source of the
    mag, mag_err, and mag_band.  Codes are the same as listed for
    There is no default value.  
  
4    COLOR1 - float variable expressing the cataloged or estimated B-V color,
    used for mag_aca, in mag.  If no colors are available, the default
    value is 0.7000.  If Tycho-2 data show colors redder than
    (B-V)=1.5, then the default value is 1.5000.  True cataloged
    color values are stored in COLOR2.

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
  
4    XREF_ID1 - long integer which is the star number in the
    AGASC Version 1.0 (= GSC1.1).  This is not a unique identifier.
     Default value of -9999.

4    XREF_ID2 - long integer which maps the entry to that in the PPM.  
    Default value of -9999.
  
4    XREF_ID3 - long integer which maps the entry to that in the Tycho Output
    Catalog (TYC2).  Default value of -9999.
  
4    XREF_ID4 - long integer which maps the entry to that in the Tycho Output
    Catalog (TYC3).  Default value of -9999.
  
4    XREF_ID5 - long integer which maps the entry to that in a future
    catalog.      Default value of -9999.
  
2    RSV4 - short integer reserved for future use.  Default value of -9999.

2    RSV5 - short integer reserved for future use.  Default value of -9999.
 
2    RSV6 - short integer reserved for future use.  Default value of -9999.


			4. ACKNOWLEDGEMENTS

The AXAF Guide and Acquisition Star Catalog version 1.7 was prepared
from AGASC1.6 and AAVSO (American Association of Variable Star Observers)
Photometric All-Sky Survey (APASS) DR9, primarily by Malgosia Sobolewska,
Tom Aldcroft, and Jean Connelly. Thanks to the entire Star Selection and
Aspect Working Group for its input in the development and testing of this
catalog. The Chandra X-ray Center is supported through NASA Contract
NAS8-39073. Information about Chandra and the Chandra X-ray Observatory
Center may be found on the WWW at http://chandra.harvard.edu/ 
Detailed information about the catalog and its construction can be
obtained from the Chandra aspect web page at
https://cxc.harvard.edu/mta/ASPECT/agasc1p7 or by emailing:
aspect_help@cfa.harvard.edu
