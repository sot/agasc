#!/usr/bin/env python

"""
Produce reports of the magnitude supplement.
"""

import os
import argparse
import numpy as np
from astropy import table, time, units as u
from cxotime import CxoTime

from agasc.supplement.magnitudes import mag_estimate_report


def parser():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument('--start', help='Time to start processing new observations.'
                                       ' CxoTime-compatible time stamp.')
    parse.add_argument('--stop', help='Time to stop processing new observations.'
                                      ' CxoTime-compatible time stamp.')
    parse.add_argument('--out-dir', default=f'./mag_stats_report',
                       help='Output directory')
    parse.add_argument('--obs-stats', default=f'mag_stats_obsid.fits',
                       help='FITS file with mag-stats for all observations')
    parse.add_argument('--agasc-stats', default=f'mag_stats_agasc.fits',
                       help='FITS file with mag-stats for all observed AGASC stars')
    parse.add_argument('--weekly-report', help="Add links to navigate weekly reports",
                       action='store_true', default=False)
    return parse


def main():
    args = parser().parse_args()

    agasc_stats = table.Table.read(args.agasc_stats)
    agasc_stats.convert_bytestring_to_unicode()
    obs_stats = table.Table.read(args.obs_stats)
    obs_stats.convert_bytestring_to_unicode()

    args.start = CxoTime(args.start)
    args.stop = CxoTime(args.stop)

    t = (obs_stats['mp_starcat_time'])
    ok = (t < args.stop) & (t > args.start) & ~obs_stats['obs_ok']
    stars = np.unique(obs_stats[ok]['agasc_id'])
    sections = [{
        'id': 'stars',
        'title': 'Stars',
        'stars': stars
    }]

    agasc_stats = agasc_stats[np.in1d(agasc_stats['agasc_id'], stars)]

    if args.weekly_report:
        t = CxoTime(args.stop)
        directory = os.path.join(args.out_dir, f'{t.date[:8]}')
        week = time.TimeDelta(7*u.day)
        nav_links = {
            'previous': f'../{(t - week).date[:8]}',
            'up': '..',
            'next': f'../{(t + week).date[:8]}'
        }
    else:
        directory = args.out_dir
        nav_links = None

    msr = mag_estimate_report.MagEstimateReport(agasc_stats,
                                                obs_stats,
                                                directory=directory)
    msr.multi_star_html(sections=sections, tstart=args.start, tstop=args.stop,
                        filename='index.html',
                        nav_links=nav_links)


if __name__ == '__main__':
    main()
