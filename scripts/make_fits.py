import os
from pathlib import Path
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import argparse


NEW_COMMENT = (
    ' ',
    '              THE AXAF GUIDE and ACQUISITION STAR CATALOG V1.7',
    ' ',
    '               An all-sky astrometric and photometric catalog',
    '                   prepared for the operation of AXAF.',
    ' ',
    'This file contains data for one of the 9537 regions constituting',
    'the AXAF Guide and Acquisition Star Catalog (AGASC) version 1.7',
    'Additional information on the AGASC may be obtained from the',
    'AXAF Science Center, Smithsonian Astrophysical Observatory,',
    '60 Garden St., Cambridge, MA 02138, or in tables elsewhere',
    'in this set of volumes (see the "comments" file).',
    'Null values are listed as -9999.',
    ' ',
    'V1.7 changes values for MAG_ACA, MAG_ACA_ERR, RSV1, RSV2 and RSV3',
    ' ',
    ' ',
    'File written on Sept 5 2018',
    ' ',
)


def get_options():
    parser = argparse.ArgumentParser(
        description="Make new AGASC fits files from old AGASC and new h5 source"
    )
    parser.add_argument("--h5", default='/proj/sot/ska/data/agasc/agasc1p7.h5')
    parser.add_argument("--olddir", default="/data/agasc1p6/agasc")
    parser.add_argument("--out", default="./testagasc")
    args = parser.parse_args()
    return args


def update_file(src, dest, newdata):
    print(src, dest)
    hdu = fits.open(src)
    hdr = hdu[0].header
    # Update AGASC VERSION
    hdr['EXTVER'] = 17

    for row in hdu[1].data:
        # Set RSV1 to be -9999.0 by default (for most stars this is a no-op, already -9999.0)
        row['RSV1'] = -9999.0
        # Find the index of the AGASC ID we have in the sorted table of newdata
        idx = np.searchsorted(newdata['AGASC_ID'], row['AGASC_ID'])
        if newdata['AGASC_ID'][idx] != row['AGASC_ID']:
            idx = np.flatnonzero(newdata['AGASC_ID'] == row['AGASC_ID'])[0]
        nrow = newdata[idx]
        if nrow['RSV3'] != 1.0:
            continue
        row['MAG_ACA'] = nrow['MAG_ACA']
        row['MAG_ACA_ERR'] = nrow['MAG_ACA_ERR']
        row['RSV1'] = nrow['RSV1']
        row['RSV2'] = nrow['RSV2']
        row['RSV3'] = nrow['RSV3']

    # Set all previous lines to a colon (replacing with a space doesn't
    # seem to actually work)
    hdr['comment'][0:19] = ":"

    for idx, line in enumerate(NEW_COMMENT):
        hdr['comment'][idx] = line

    # Update table comments for RSV1 RSV2 RSV3
    hdu[1].header.comments["TTYPE26"] = "APASS V - i magnitude (COLOR3)"
    hdu[1].header.comments["TTYPE27"] = "APASS V magnitude (units of 0.001mags) "
    # I'd actually like to add TUNIT27, but that changes all the columns more than I'd like
    hdu[1].header.comments["TTYPE28"] = "Mag updated for AGASC 1.7 (1=Yes, 0=No) "

    hdu.writeto(dest)


def main(h5file, srcdir, outdir):
    # This sorts the full agasc in memory, so don't do it on a puny machine
    tbl = Table.read(h5file)
    tbl.sort('AGASC_ID')
    # redefine outdir as the agasc subdirectory in outdir
    outdir = os.path.join(outdir, 'agasc')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for topdir in Path(srcdir).glob("*"):
        newdir = Path(outdir).joinpath(topdir.name)
        if not Path(newdir).exists():
            newdir.mkdir()
        for f in topdir.glob("*.fit"):
            newfile = newdir.joinpath(f.name)
            update_file(f, newfile, tbl)


if __name__ == '__main__':
    args = get_options()
    main(args.h5, args.olddir, args.out)
