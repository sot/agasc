import os
import jinja2
from agasc.supplement.magnitudes import mag_estimate
from astropy import table
import numpy as np
from cxotime import CxoTime
import matplotlib.pyplot as plt

from email.mime.text import MIMEText
from subprocess import Popen, PIPE
import platform
import getpass


class MagEstimateReport:
    def __init__(self, agasc_stats, obs_stats, directory='./mag_estimates_reports'):
        self.agasc_stats = agasc_stats
        self.obs_stats = obs_stats
        self.directory = directory

    def single_star_html(self, agasc_id, directory,
                         static_dir='https://cxc.cfa.harvard.edu/mta/ASPECT/www_resources',
                         highlight_obs=lambda o: ~o['obs_ok']):
        if np.sum(self.agasc_stats['agasc_id'] == agasc_id) == 0:
            return

        star_template = jinja2.Template(STAR_REPORT_BOOTSTRAP)

        if not os.path.exists(directory):
            os.makedirs(directory)

        o = self.obs_stats[self.obs_stats['agasc_id'] == agasc_id]
        if len(o) == 0:
            raise Exception(f'agasc_id {agasc_id} has not observations')
        o.sort(keys=['mp_starcat_time'])
        s = self.agasc_stats[self.agasc_stats['agasc_id'] == agasc_id][0]
        s = {k: s[k] for k in s.colnames}
        s['n_obs_bad'] = s['n_obsids'] - s['n_obsids_ok'] - s['n_obsids_fail']
        s['last_obs'] = ':'.join(o[-1]['mp_starcat_time'].split(':')[:4])

        # OBSIDs can be repeated
        obsids = list(np.unique(o[highlight_obs(o)]['obsid']))

        args = [{'only_ok': False, 'draw_agasc_mag': True, 'draw_legend': True, 'ylim': 'max'},
                {'title': 'Magnitude Estimates',
                 'only_ok': True,
                 'ylim': 'stats',
                 'highlight_obsid': obsids,
                 'draw_obs_mag_stats': True,
                 'draw_agasc_mag_stats': True,
                 'draw_legend': True,
                 'outside_markers': True
                 },
                {'type': 'flags'}]
        for obsid in obsids:
            args.append({'obsid': obsid,
                         'ylim': 'fit',
                         'only_ok': False,
                         'draw_obs_mag_stats': True,
                         'draw_agasc_mag_stats': True,
                         'draw_legend': True,
                         'draw_roll_mean': True,
                         'outside_markers': True
                         })
            args.append({'type': 'flags', 'obsid': obsid})
        fig = self.plot_set(agasc_id, args=args, filename=os.path.join(directory, f'mag_stats.png'))
        plt.close(fig)

        with open(os.path.join(directory, 'index.html'), 'w') as out:
            out.write(star_template.render(agasc_stats=s,
                                           obs_stats=o.as_array(),
                                           static_dir=static_dir))
        return os.path.join(directory, 'index.html')

    def multi_star_html(self, sections=None, updated_stars=None, fails=(),
                        tstart=None, tstop=None, report_date=None,
                        filename=None,
                        include_all_stars=False,
                        make_single_reports=True,
                        nav_links=None,
                        highlight_obs=lambda o: ~o['obs_ok'],
                        static_dir='https://cxc.cfa.harvard.edu/mta/ASPECT/www_resources'):
        if sections is None:
            sections = []
        run_template = jinja2.Template(RUN_REPORT_SIMPLE)

        updated_star_ids = \
            updated_stars['agasc_id'] if updated_stars is not None and len(updated_stars) else []
        if updated_stars is None:
            updated_stars = []

        info = {
            'tstart': tstart if tstart else CxoTime(self.obs_stats['mp_starcat_time']).min().date,
            'tstop': tstop if tstop else CxoTime(self.obs_stats['mp_starcat_time']).max().date,
            'report_date': report_date if report_date else CxoTime.now().date
        }

        if filename is None:
            filename = f'mag_estimates_{info["tstart"]}-{info["tstop"]}.html'

        # this is the list of agasc_id for which we will generate individual reports (if possible)
        if sections:
            agasc_ids = np.concatenate([np.array(s['stars'], dtype=int) for s in sections])
        else:
            agasc_ids = []
        if include_all_stars:
            sections.append({
                'id': 'other_stars',
                'title': 'Other Stars',
                'stars': self.agasc_stats['agasc_id'][~np.in1d(self.agasc_stats['agasc_id'], agasc_ids)]
            })
            agasc_ids = self.agasc_stats['agasc_id']
        failed_agasc_ids = [f['agasc_id'] for f in fails
                            if f['agasc_id'] and int(f['agasc_id']) in self.obs_stats['agasc_id']]
        agasc_ids = np.unique(np.concatenate([agasc_ids, failed_agasc_ids]))

        # this turns all None into '' in a new list of failures
        fails = [{k: '' if v is None else v for k, v in f.items()} for i, f in enumerate(fails)]

        # check how many observations were added in this run, and how many of those are ok
        new_obs = self.obs_stats[(self.obs_stats['mp_starcat_time'] >= info["tstart"]) &
                            (self.obs_stats['mp_starcat_time'] <= info["tstop"])]. \
            group_by('agasc_id')[['agasc_id', 'obsid', 'obs_ok']]. \
            groups.aggregate(np.count_nonzero)[['agasc_id', 'obsid', 'obs_ok']]
        new_obs['n_obs_bad_new'] = new_obs['obsid'] - new_obs['obs_ok']

        # add some extra fields
        agasc_stats = self.agasc_stats.copy()
        if len(agasc_stats) == 0:
            return [], []
        all_agasc_ids = np.unique(np.concatenate([
            new_obs['agasc_id'],
            [f['agasc_id'] for f in fails]
        ]))

        assert np.all(np.in1d(agasc_stats['agasc_id'], all_agasc_ids)), 'Not all AGASC IDs are in new obs.'
        agasc_stats['n_obs_bad'] = agasc_stats['n_obsids'] - agasc_stats['n_obsids_ok']
        agasc_stats['flag'] = '          '
        if len(agasc_stats):
            agasc_stats = table.join(agasc_stats, new_obs[['agasc_id', 'n_obs_bad_new']],
                                     keys=['agasc_id'])
        tooltips = {
            'warning': 'At least one bad observation',
            'danger': 'At least one new bad observation'
        }
        agasc_stats['flag'][:] = ''
        agasc_stats['flag'][agasc_stats['n_obs_bad'] > 0] = 'warning'
        agasc_stats['flag'][agasc_stats['n_obs_bad_new'] > 0] = 'danger'
        agasc_stats['delta'] = (agasc_stats['t_mean_dr3'] - agasc_stats['mag_aca'])
        agasc_stats['sigma'] = (agasc_stats['t_mean_dr3'] - agasc_stats['mag_aca'])/agasc_stats['mag_aca_err']
        agasc_stats['new'] = True
        agasc_stats['new'][np.in1d(agasc_stats['agasc_id'], updated_star_ids)] = False
        agasc_stats['update_mag_aca'] = np.nan
        agasc_stats['update_mag_aca_err'] = np.nan
        agasc_stats['last_obs'] = CxoTime(agasc_stats['last_obs_time']).date
        if len(updated_stars):
            agasc_stats['update_mag_aca'][np.in1d(agasc_stats['agasc_id'], updated_star_ids)] = \
                updated_stars['mag_aca']
            agasc_stats['update_mag_aca_err'][np.in1d(agasc_stats['agasc_id'], updated_star_ids)] =\
                updated_stars['mag_aca_err']
        # make all individual star reports
        star_reports = {}
        for agasc_id in np.atleast_1d(agasc_ids):
            try:
                dirname = os.path.join(self.directory,
                                       'stars',
                                       f'{agasc_id//1e7:03.0f}',
                                       f'{agasc_id:.0f}')
                if make_single_reports:
                    self.single_star_html(
                        agasc_id,
                        directory=dirname,
                        highlight_obs=highlight_obs
                    )
                if os.path.exists(dirname):
                    star_reports[agasc_id] = dirname
            except mag_estimate.MagStatsException:
                pass

        # remove empty sections, and set the star tables for each of the remaining sections
        sections = sections.copy()
        sections = [section for section in sections if len(section['stars'])]
        for section in sections:
            section['stars'] = agasc_stats[np.in1d(agasc_stats['agasc_id'],
                                                   section['stars'])].as_array()

        # this is a hack
        star_reports = {i: os.path.relpath(star_reports[i], self.directory) for i in star_reports}
        # make report
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        with open(os.path.join(self.directory, filename), 'w') as out:
            out.write(run_template.render(info=info,
                                          sections=sections,
                                          failures=fails,
                                          star_reports=star_reports,
                                          nav_links=nav_links,
                                          tooltips=tooltips,
                                          static_dir=static_dir))

    def plot_agasc_id_single(self, agasc_id, obsid=None,
                             telem=None,
                             highlight_obsid=(),
                             highlight_outliers=True,
                             only_ok=True,
                             title=None,
                             draw_agasc_mag_stats=False,
                             draw_obs_mag_stats=False,
                             draw_agasc_mag=False,
                             draw_roll_mean=False,
                             draw_legend=False,
                             ylim='fit',
                             ax=None,
                             outside_markers=False):
        if title is not None:
            ax.set_title(title)
        if type(highlight_obsid) is not list and np.isscalar(highlight_obsid):
            highlight_obsid = [highlight_obsid]

        agasc_stat = self.agasc_stats[self.agasc_stats['agasc_id'] == agasc_id][0]
        obs_stats = self.obs_stats[self.obs_stats['agasc_id'] == agasc_id]
        arg_obsid = obsid
        if arg_obsid:
            obs_stats = obs_stats[obs_stats['obsid'] == arg_obsid]

        previous_axes = plt.gca()
        if ax is not None:
            plt.sca(ax)
        if ax is None:
            ax = plt.gca()
        if telem is None:
            telem = mag_estimate.get_telemetry_by_agasc_id(agasc_id, ignore_exceptions=True)
            telem = mag_estimate.add_obs_info(telem, obs_stats)

        obsids = [arg_obsid] if arg_obsid else np.unique(telem['obsid'])

        timeline = telem[['times', 'mags', 'obsid', 'obs_outlier']].copy()
        timeline['index'] = np.arange(len(timeline))
        timeline['mean'] = np.nan
        timeline['std'] = np.nan
        timeline['mag_mean'] = np.nan
        timeline['mag_std'] = np.nan
        for i, obsid in enumerate(np.unique(timeline['obsid'])):
            sel = (obs_stats['obsid'] == obsid)
            if draw_obs_mag_stats and np.any(sel):
                timeline['mag_mean'][timeline['obsid'] == obsid] = obs_stats[sel]['mean'][0]
                timeline['mag_std'][timeline['obsid'] == obsid] = obs_stats[sel]['std'][0]

        timeline = timeline.as_array()

        ok = (telem['AOACASEQ'] == 'KALM') & (telem['AOACIIR'] == 'OK') & \
             (telem['AOACISP'] == 'OK') & (telem['AOPCADMD'] == 'NPNT')
        ok = ok & (telem['dr'] < 5)

        # set the limits of the plot beforehand
        ok_ylim = ok & (telem['mags'] > 0)
        ylims_set = False
        if arg_obsid is not None:
            ok_ylim = ok_ylim & (telem['obsid'] == arg_obsid)
        if ylim == 'fit':
            if np.sum(ok_ylim):
                q25, q50, q75 = np.quantile(telem['mags'][ok_ylim], [.25, 0.5, 0.75])
            else:
                q25, q50, q75 = np.quantile(telem['mags'], [.25, 0.5, 0.75])
            iqr = max(q75 - q25, 0.05)
            ax.set_ylim((q25 - 2 * iqr, q75 + 2 * iqr))
            ylims_set = True
        elif ylim == 'stats':
            if arg_obsid is not None:
                q25, q50, q75 = obs_stats[['q25', 'median', 'q75']][0]
                iqr = max(q75 - q25, 0.05)
                ax.set_ylim((q25 - 3 * iqr, q75 + 3 * iqr))
                ylims_set = True
            elif arg_obsid is None and agasc_stat['mag_obs_std'] > 0:
                ylim = (agasc_stat['mag_obs'] - 6 * agasc_stat['mag_obs_std'],
                        agasc_stat['mag_obs'] + 6 * agasc_stat['mag_obs_std'])
                ax.set_ylim(ylim)
                ylims_set = True
        if ylim == 'max' or not ylims_set:
            if np.any(ok_ylim):
                ymin, ymax = np.min(telem['mags'][ok_ylim]), np.max(telem['mags'][ok_ylim])
            else:
                ymin, ymax = np.min(telem['mags']), np.max(telem['mags'])
            dy = max(0.3, ymax - ymin)
            ax.set_ylim((ymin - 0.1 * dy, ymax + 0.1 * dy))

        # set flags for different categories of markers
        highlighted = np.zeros(len(timeline['times']), dtype=bool)
        if highlight_obsid:
            highlighted = highlighted | np.in1d(timeline['obsid'], highlight_obsid)
        if highlight_outliers:
            highlighted = highlighted | timeline['obs_outlier']

        ymin, ymax = ax.get_ylim()
        ymin, ymax = (ymin + 0.01 * (ymax - ymin)), (ymax - 0.01 * (ymax - ymin))
        top = np.ones_like(timeline['mags']) * ymax
        bottom = np.ones_like(timeline['mags']) * ymin
        if outside_markers:
            outside_up = timeline['mags'] >= ymax
            outside_down = timeline['mags'] <= ymin
        else:
            outside_up = np.zeros_like(timeline['mags'], dtype=bool)
            outside_down = np.zeros_like(timeline['mags'], dtype=bool)
        inside = ~outside_up & ~outside_down

        # loop over obsids, making each plot
        marker_handles = []
        line_handles = []
        limits = {}
        for i, obsid in enumerate(obsids):
            in_obsid = timeline['obsid'] == obsid
            limits[obsid] = (timeline['index'][timeline['obsid'] == obsid].min(),
                             timeline['index'][timeline['obsid'] == obsid].max())
            if not only_ok and np.any(in_obsid & ~ok):
                s = plt.scatter(timeline['index'][in_obsid & ~ok & inside],
                                timeline[in_obsid & ~ok & inside]['mags'],
                                s=10, marker='.', color='r', label='not OK')
                if i == 0:
                    marker_handles.append(s)
                _ = plt.scatter(
                    timeline['index'][in_obsid & ~ok & outside_down],
                    bottom[in_obsid & ~ok & outside_down],
                    s=10, marker='v', color='r'
                )
                _ = plt.scatter(
                    timeline['index'][in_obsid & ~ok & outside_up],
                    top[in_obsid & ~ok & outside_up],
                    s=10, marker='^', color='r'
                )

            if np.any(in_obsid & ok & highlighted):
                s = plt.scatter(
                    timeline['index'][in_obsid & ok & highlighted & inside],
                    timeline[in_obsid & ok & highlighted & inside]['mags'],
                    s=10, marker='.', color='orange', label='Highlighted'
                )
                if i == 0:
                    marker_handles.append(s)
                _ = plt.scatter(
                    timeline['index'][in_obsid & ok & highlighted & outside_up],
                    top[in_obsid & ok & highlighted & outside_up],
                    s=10, marker='^', color='orange', label='Highlighted'
                )
                _ = plt.scatter(
                    timeline['index'][in_obsid & ok & highlighted & outside_down],
                    bottom[in_obsid & ok & highlighted & outside_down],
                    s=10, marker='v', color='orange', label='Highlighted'
                )

            if np.any(in_obsid & ok & ~highlighted):
                s = plt.scatter(
                    timeline['index'][in_obsid & ok & ~highlighted & inside],
                    timeline[in_obsid & ok & ~highlighted & inside]['mags'],
                    s=10, marker='.', color='k', label='OK'
                )
                if i == 0:
                    marker_handles.append(s)
                _ = plt.scatter(
                    timeline['index'][
                        in_obsid & ok & (~highlighted) & outside_down],
                    bottom[in_obsid & ok & ~highlighted & outside_down],
                    s=10, marker='v', color='k', label='OK'
                )
                _ = plt.scatter(
                    timeline['index'][in_obsid & ok & ~highlighted & outside_up],
                    top[in_obsid & ok & ~highlighted & outside_up],
                    s=10, marker='^', color='k', label='OK'
                )
            sel = (obs_stats['obsid'] == obsid)
            if draw_obs_mag_stats and np.sum(sel):
                label = '' if i else 'mag$_{OBSID}$'
                if (np.isfinite(obs_stats[sel][0]['t_mean']) and
                        np.isfinite(obs_stats[sel][0]['t_std'])):
                    mag_mean = obs_stats[sel][0]['t_mean']
                    mag_mean_minus = mag_mean - obs_stats[sel][0]['t_std']
                    mag_mean_plus = mag_mean + obs_stats[sel][0]['t_std']
                    lh = ax.plot(
                        limits[obsid],
                        [mag_mean, mag_mean],
                        linewidth=2, color='orange', label=label
                    )
                    if i == 0:
                        line_handles += lh
                    ax.fill_between(limits[obsid],
                                    [mag_mean_minus, mag_mean_minus],
                                    [mag_mean_plus, mag_mean_plus],
                                    color='orange', alpha=0.1, zorder=100)
                else:
                    (
                        ax.plot([], [], linewidth=2, color='orange', label=label)
                    )
            if draw_roll_mean:
                o = (timeline['obsid'] == obsid) & ok

                roll_mean = mag_estimate.rolling_mean(timeline['times'],
                                                      timeline['mags'],
                                                      window=100,
                                                      selection=o)
                lh = ax.plot(
                    timeline['index'], roll_mean, '--',
                    linewidth=1, color='r', label='rolling mean'
                )
                if i == 0:
                    line_handles += lh

        sorted_obsids = sorted(limits.keys(), key=lambda l: limits[l][1])
        for i, obsid in enumerate(sorted_obsids):
            (tmin, tmax) = limits[obsid]
            ax.plot([tmin, tmin], ax.get_ylim(), ':', color='purple', scaley=False)
            shift = 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * (1 + i % 3)
            ax.text(np.mean([tmin, tmax]), ax.get_ylim()[0] + shift, f'{obsid}',
                    verticalalignment='top', horizontalalignment='center')
        if limits:
            tmax = max([v[1] for v in limits.values()])
            ax.plot([tmax, tmax], ax.get_ylim(), ':', color='purple', scaley=False)

        xlim = ax.get_xlim()
        ax.set_xlim(xlim)

        if draw_agasc_mag:
            mag_aca = np.mean(agasc_stat['mag_aca'])
            line_handles += (
                ax.plot(xlim, [mag_aca, mag_aca], label='mag$_{AGASC}$',
                        color='green', scalex=False, scaley=False)
            )

        if (draw_agasc_mag_stats and
                np.isfinite(agasc_stat['mag_obs']) and agasc_stat['mag_obs'] > 0):
            mag_weighted_mean = agasc_stat['mag_obs']
            mag_weighted_std = agasc_stat['mag_obs_std']
            line_handles += (
                ax.plot(ax.get_xlim(), [mag_weighted_mean, mag_weighted_mean],
                        label='mag', color='r', scalex=False)
            )
            ax.fill_between(xlim,
                            [mag_weighted_mean - mag_weighted_std,
                             mag_weighted_mean - mag_weighted_std],
                            [mag_weighted_mean + mag_weighted_std,
                             mag_weighted_mean + mag_weighted_std],
                            color='r', alpha=0.1)

        if draw_legend:
            ax.set_xlim((xlim[0], xlim[1] + 0.1 * (xlim[1] - xlim[0])))
            if marker_handles:
                ax.add_artist(plt.legend(handles=marker_handles, loc='lower right'))
            if line_handles:
                plt.legend(handles=line_handles, loc='upper right')

        plt.sca(previous_axes)

    @staticmethod
    def plot_flags(telemetry, ax=None, obsid=None):
        if ax is None:
            ax = plt.gca()

        timeline = telemetry[['times', 'mags', 'obsid', 'obs_ok', 'dr', 'AOACFCT',
                              'AOACASEQ', 'AOACIIR', 'AOACISP', 'AOPCADMD',
                              ]]
        timeline['x'] = np.arange(len(timeline))
        timeline['y'] = np.ones(len(timeline))
        timeline = timeline.as_array()

        if obsid:
            timeline = timeline[timeline['obsid'] == obsid]

        obsids = np.unique(timeline['obsid'])

        all_ok = ((timeline['AOACASEQ'] == 'KALM') &
                  (timeline['AOPCADMD'] == 'NPNT') &
                  (timeline['AOACFCT'] == 'TRAK') &
                  (timeline['AOACIIR'] == 'OK') &
                  (timeline['AOACISP'] == 'OK') &
                  (timeline['dr'] < 3)
                  )
        flags = [
            ('dr > 5', ((timeline['AOACASEQ'] == 'KALM') &
                        (timeline['AOPCADMD'] == 'NPNT') &
                        (timeline['AOACFCT'] == 'TRAK') &
                        (timeline['dr'] >= 5))),
            ('Ion. rad.', (timeline['AOACIIR'] != 'OK')),
            ('Sat. pixel.', (timeline['AOACISP'] != 'OK')),
            ('not track', ((timeline['AOACASEQ'] == 'KALM') &
                           (timeline['AOPCADMD'] == 'NPNT') &
                           (timeline['AOACFCT'] != 'TRAK'))),
            ('not Kalman', ((timeline['AOACASEQ'] != 'KALM') |
                            (timeline['AOPCADMD'] != 'NPNT'))),
        ]

        if obsid is None:
            all_ok = timeline['obs_ok'] & all_ok
            flags = [('OBS not OK', ~timeline['obs_ok'])] + flags

        limits = {}
        for i, obsid in enumerate(obsids):
            limits[obsid] = (timeline['x'][timeline['obsid'] == obsid].min(),
                             timeline['x'][timeline['obsid'] == obsid].max())

        ok = [f[1] for f in flags]
        labels = [f[0] for f in flags]
        ticks = [i for i in range(len(flags))]

        for i in range(len(ok)):
            ax.scatter(timeline['x'][ok[i]], ticks[i] * timeline['y'][ok[i]], s=4, marker='.',
                       color='k')
        ax.set_yticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_ylim((-1, ticks[-1] + 1))
        ax.grid(True, axis='y', linestyle=':')

        sorted_obsids = sorted(limits.keys(), key=lambda l: limits[l][1])
        for i, obsid in enumerate(sorted_obsids):
            (tmin, tmax) = limits[obsid]
            ax.axvline(tmin, linestyle=':', color='purple')
            if i == len(limits) - 1:
                ax.axvline(tmax, linestyle=':', color='purple')

        dr = timeline['dr'].copy()
        dr[dr > 10] = 10
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_dr = divider.append_axes("bottom", size='25%', pad=0., sharex=ax)
        ax_dr.set_ylabel('dr')
        ax_dr.scatter(
            timeline['x'][all_ok & (dr < 10)], dr[all_ok & (dr < 10)],
            s=3, marker='.', color='k'
        )
        ax_dr.scatter(
            timeline['x'][all_ok & (dr >= 10)], dr[all_ok & (dr >= 10)],
            s=3, marker='^', color='k'
        )
        ax_dr.scatter(
            timeline['x'][~all_ok & (dr < 10)], dr[~all_ok & (dr < 10)],
            s=3, marker='.', color='r'
        )
        ax_dr.scatter(
            timeline['x'][~all_ok & (dr >= 10)],
            dr[~all_ok & (dr >= 10)],
            s=3, marker='^',
            color='r'
        )
        ax_dr.set_ylim((-0.5, 10.5))
        ax_dr.set_yticks([0., 2.5, 5, 7.5, 10], minor=True)
        ax_dr.set_yticks([0., 5, 10], minor=False)
        ax_dr.grid(True, axis='y', linestyle=':')

        for i, obsid in enumerate(sorted_obsids):
            (tmin, tmax) = limits[obsid]
            ax_dr.axvline(tmin, linestyle=':', color='purple')
            if i == len(sorted_obsids) - 1:
                ax_dr.axvline(tmax, linestyle=':', color='purple')

    def plot_set(self, agasc_id, args, telem=None, filename=None):
        if not args:
            return
        if telem is None:
            telem = mag_estimate.get_telemetry_by_agasc_id(agasc_id, ignore_exceptions=True)
            telem = mag_estimate.add_obs_info(
                telem,
                self.obs_stats[self.obs_stats['agasc_id'] == agasc_id]
            )
        fig, ax = plt.subplots(len(args), 1, figsize=(15, 3.5 * len(args)))
        if len(args) == 1:
            ax = [ax]
        ax[0].set_title(f'AGASC ID {agasc_id}')

        for i, kwargs in enumerate(args):
            if 'type' in kwargs and kwargs['type'] == 'flags':
                o = kwargs['obsid'] if 'obsid' in kwargs else None
                self.plot_flags(telem, ax[i], obsid=o)
                if i:
                    ax[i].set_xlim(ax[i - 1].get_xlim())
            else:
                self.plot_agasc_id_single(agasc_id, telem=telem, ax=ax[i], **kwargs)

        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename)

        return fig


RUN_REPORT_SIMPLE = """<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet"
          href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
          integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
          crossorigin="anonymous">
  </head>
  <body>

    <div class="container">
    {% if nav_links %}
    <!--
    <nav aria-label="Page navigation example">
      <ul class="pagination">
        <li class="page-item">
          <a class="page-link" href='{{ nav_links.previous}}'> Previous </a>
        </li>
        <li class="page-item">
          <a class="page-link" href='{{ nav_links.up}}'> Up </a>
        </li>
        <li class="page-item">
          <a class="page-link" href='{{ nav_links.next}}'> Next </a>
        </li>
      </ul>
    </nav>
    -->
    <nav aria-label="Page navigation example">
      <ul class="pagination">
        <li class="page-item"><a class="page-link" href='{{ nav_links.previous}}'>
          <span aria-hidden="true">&laquo;</span>
          <span class="sr-only">Previous</span>
        </a></li>
        <li class="page-item"><a class="page-link" href='{{ nav_links.up}}'> 
          <!--span aria-hidden="true">&#8896;</span-->
          <!--span aria-hidden="true">&Hat;</span-->
          <!--span aria-hidden="true">&#8962;</span-->
          <span aria-hidden="true">&#127968;</span>
          <span class="sr-only">Up</span>
        </a></li>
        <li class="page-item"><a class="page-link" href='{{ nav_links.next}}'>
          <span aria-hidden="true">&raquo;</span>
          <span class="sr-only">Next</span>
        </a></li>
      </ul>
    </nav>

    {% endif %}
    <h1> ACA Magnitude Statistics </h1>
    <h2> {{ info.report_date }} Update Report </h2>
    <table class="table table-sm">
      <tr>
        <td style="width: 50%"> Time range </td>
        <td style="width: 50%"> {{ info.tstart }} &ndash; {{ info.tstop }} </td>
      </tr>
      {%- for section in sections %}
      <tr>
        <td> <a href="#{{ section.id }}"> {{ section.title }} </a> </td>
        <td> {{ section.stars | length }} </td>
      </tr>
      {%- endfor %}    
      <tr>
        <td> {% if failures -%} <a href="#failures"> Failures </a>
             {%- else -%} Failures {%- endif %} </td>
        <td> {{ failures | length }} </td>
      </tr>
    </table>

    {%- for section in sections %}
    <a name="{{ section.id }}"> </a>
    <h3> {{ section.title }} </h3>
    <table class="table table-hover">
      <tr>
      <tr>
        <th data-toggle="tooltip" data-placement="top" title="ID in AGASC"> AGASC ID </th>
        <th data-toggle="tooltip" data-placement="top" title="Last time the star was observed"> Last Obs </th>
        <th data-toggle="tooltip" data-placement="top" title="Number of times the star has been observed"> n<sub>obs</sub> </th>
        <th data-toggle="tooltip" data-html="true" data-placement="top" title="Observations not included in calculation <br/> n &gt; 10 <br/>f_ok &gt; 0.3 <br/> &langle; &delta; <sub>mag</sub> &rangle; <sub>100s</sub>  < 1"> n<sub>bad</sub> </th>
        <th data-toggle="tooltip" data-html="true" data-placement="top" title="New observations not included in calculation <br/> n &gt; 10 <br/>f_ok &gt; 0.3 <br/> &langle; &delta; <sub>mag</sub> &rangle; <sub>100s</sub>  < 1"> n<sub>bad new</sub> </th>
        <th data-toggle="tooltip" data-placement="top" data-html="true" title="tracking time as fraction of total time: <br/> AOACASEQ == 'KALM' <br/> AOACIIR == 'OK' <br/> AOACISP == 'OK' <br/> AOPCADMD == 'NPNT' <br/> AOACFCT == 'TRAK' <br/> OBS_OK)"> f<sub>track</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Fraction of the tracking time within 5 arcsec of target"> f<sub>5 arcsec</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Magnitude in AGASC"> mag<sub>catalog</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Magnitude observed"> mag<sub>obs</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Difference between observed and catalog magnitudes"> &delta;<sub>mag cat</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Difference between observed and catalog magnitudes, divided by catalog magnitude error"> &delta;<sub>mag</sub>/&sigma;<sub>mag</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Variation in observed magnitude from the last version of AGASC supplement"> &delta;<sub>mag</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Variation in observed magnitude standard deviation from the last version of AGASC supplement"> &delta;<sub>&sigma;</sub> </th>
        <th data-toggle="tooltip" data-placement="top" title="Color in AGASC"> color </th>
      </tr>
      {%- for star in section.stars %}
      <tr {% if star.flag != '' -%}
          class="table-{{ star.flag }}"
          data-toggle="tooltip"
          data-placement="top" title="{{ tooltips[star.flag] }}"
        {%- endif -%}
        >
        <td>
        {%- if star.agasc_id in star_reports -%}
          <a href="{{ star_reports[star.agasc_id] }}/index.html"> {{ star.agasc_id }} </a>
        {%- else -%}
          {{ star.agasc_id }}
        {%- endif -%}
        </td>
        <td> {{ star.last_obs[:8] }} </td>
        <td> {{ star.n_obsids }}  </td>
        <td> {%- if star.n_obs_bad > 0 %} {{ star.n_obs_bad }} {% endif %} </td>
        <td> {%- if star.n_obs_bad > 0 %} {{ star.n_obs_bad_new }} {% endif %} </td>
        <td> {{ "%.1f" | format(100*star.f_ok) }}%  </td>
        <td> {{ "%.1f" | format(100*star.f_dr5) }}% </td>
        <td {% if star.selected_mag_aca_err -%}
              class="table-info"
              data-toggle="tooltip" data-placement="top"
              title="Large magnitude error in catalog"
            {%- endif %}>
          {{ "%.2f" | format(star.mag_aca) }} &#177; {{ "%.2f" | format(star.mag_aca_err) }}
        </td>
        <td>
          {{ "%.2f" | format(star.mag_obs) }} &#177; {{ "%.2f" | format(star.mag_obs_err) }}
        </td>
        <td {%- if star.selected_atol %}
              class="table-info"
              data-toggle="tooltip" data-placement="top"
              title="Large absolute difference between observed and catalogue magnitudes"
            {% endif %}> {{ "%.2f" | format(star.delta) }}  </td>
        <td {%- if star.selected_rtol %}
              class="table-info"
              data-toggle="tooltip" data-placement="top"
              title="Large relative difference between observed and catalogue magnitudes"
            {% endif %}> {{ "%.2f" | format(star.sigma) }}  </td>
        <td>
          {%- if star.new %} &ndash; {% else -%}
          {{ "%.2f" | format(star.update_mag_aca) }}{% endif -%}
        </td>
        <td>
          {%- if star.new %} &ndash; {% else -%}
          {{ "%.2f" | format(star.update_mag_aca_err) }}{% endif -%}
        </td>
        <td {%- if star.selected_color %}
              class="table-info"
              data-toggle="tooltip" data-placement="top"
              title="Color==1.5 or color==0.7 in catalog"
            {% endif %}> {{ "%.2f" | format(star.color) }}  </td>
      </tr>
      {%- endfor %}
    <table>
    {%- endfor %}

    <a name="failures"> </a>
    {%- if failures %}
    <h3> Failures </h3>
    <table class="table table-hover">
      <tr>
        <th> AGASC ID </th>
        <th> OBSID </th>
        <th> Message </th>
      </tr>
      {%- for failure in failures %}
      <tr>
        <td> {%- if failure.agasc_id in star_reports -%}
          <a href="{{ star_reports[failure.agasc_id] }}/index.html"> {{ failure.agasc_id }} </a>
          {%- else -%} {{ failure.agasc_id }} {%- endif -%} </td>
        <td> {{ failure.obsid }} </td>
        <td> {{ failure.msg }} </td>
      </tr>
      {%- endfor %}
    </table>
    {% endif %}
    </div>

  <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
    integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
    crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
    integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
    crossorigin="anonymous"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
    integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
    crossorigin="anonymous"></script>

  <script type="text/javascript">
    $(document).ready(function() {
    $("body").tooltip({ selector: '[data-toggle=tooltip]' });
});
  </script>
</html>
"""


STAR_REPORT_BOOTSTRAP = """<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet"
          href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
          integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
          crossorigin="anonymous">
  </head>
  <body>

    <div class="container">
      <h1> AGASC ID {{ agasc_stats.agasc_id }} </h1>
      <h3> Info </h3>
      <div class="row">
        <div class="col-md">
          <table class="table table-bordered table-sm">
            <tr>
              <td style="width: 30%"> Last Obs. </td>
              <td style="width: 30%"> {{ agasc_stats.last_obs }} </td>
            </tr>
            <tr>
              <td style="width: 30%"> mag<sub>catalog</sub> </td>
              <td style="width: 30%">
                {{ "%.2f" | format(agasc_stats.mag_aca) }} &#177; {{ "%.2f" | format(agasc_stats.mag_aca_err) }}
              </td>
            </tr>
            <tr>
              <td> mag<sub>3 arcsec </sub> </td>
              <td>
                {{ "%.2f" | format(agasc_stats.t_mean_dr3) }} &#177; {{ "%.2f" | format(agasc_stats.t_std_dr3) }}
              </td>
            </tr>
            <tr>
              <td> mag<sub>5 arcsec </sub> </td>
              <td>
                {{ "%.2f" | format(agasc_stats.t_mean_dr5) }} &#177; {{ "%.2f" | format(agasc_stats.t_std_dr5) }}
              </td>
            </tr>
          </table>
        </div>
        <div class="col-md">
          <table class="table table-bordered table-sm">
            <tr>
              <td> N<sub>obs</sub> </td>
              <td>
                {{ agasc_stats.n_obsids }} <span{%- if agasc_stats.n_obs_bad %} style="color:red;"{% endif -%}> ({{ agasc_stats.n_obs_bad }} bad) <span>
              </td>
            </tr>
            <tr>
              <td> f<sub>ok</sub> </td>
              <td> {{ "%.1f" | format(100*agasc_stats.f_ok) }}%  </td>
            </tr>
            <tr>
              <td> f<sub>3 arcsec</sub> </td>
              <td> {{ "%.1f" | format(100*agasc_stats.f_dr3) }}% </td>
            </tr>
            <tr>
              <td> f<sub>5 arcsec</sub> </td>
              <td> {{ "%.1f" | format(100*agasc_stats.f_dr5) }}% </td>
            </tr>
          </table>
        </div>
      </div>


      <h3> Timeline </h3>
      <img src="mag_stats.png" width="100%"/>

      <h3> Observation Info </h3>
      <table  class="table table-hover">
        <tr>
          <th data-toggle="tooltip" data-placement="top" title="OBSID"> OBSID </th>
          <th data-toggle="tooltip" data-placement="top" title="MP starcat time"> Time </th>
          <th data-toggle="tooltip" data-placement="top" title="Pixel row"> Row </th>
          <th data-toggle="tooltip" data-placement="top" title="Pixel column"> Col </th>
          <!-- th data-toggle="tooltip" data-placement="top" data-html="true" title="Observation is considered in the calculation <br/> n &gt; 10 <br/>f_ok &gt; 0.3 <br/> &langle; &delta; <sub>mag</sub> &rangle; <sub>100s</sub>  < 1"> OK </th -->
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Number of time samples"> N </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Number of time samples considered as 'tracking' <br/> AOACASEQ == 'KALM' <br/> AOACIIR == 'OK' <br/> AOACISP == 'OK' <br/> AOPCADMD == 'NPNT' <br/> AOACFCT == 'TRAK' <br/> OBS_OK)"> N<sub>ok</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Number of outlying samples"> N<sub>out</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Tracking time as a fraction of the total time <br/> AOACASEQ == 'KALM' <br/> AOACIIR == 'OK' <br/> AOACISP == 'OK' <br/> AOPCADMD == 'NPNT' <br/> AOACFCT == 'TRAK' <br/> OBS_OK)"> f<sub>track</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Fraction of tracking time within 3 arcsec of target"> f<sub>dr3</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Fraction of tracking time within 5 arcsec of target"> f<sub>dr5</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Time where slot is tracking and target within 3 arcsec <br/> as fraction of total time"> f<sub>ok</sub> </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="100-second Rolling mean of mag - &langle; mag &rangle;"> &langle; &delta; <sub>mag</sub> &rangle; <sub>100s</sub>  </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Mean magnitude"> &langle; mag &rangle; </th>
          <th data-toggle="tooltip" data-placement="top" data-html="true" title="Magnitude uncertainty"> &sigma;<sub>mag</sub> </th>
          <th> Comments </th>
        </tr>
        {%- for s in obs_stats %}
        <tr {%- if not s.obs_ok %} class="table-danger" {% endif %}>
          <td> <a href="https://web-kadi.cfa.harvard.edu/mica/?obsid_or_date={{ s.obsid }}"> {{ s.obsid }} </td>
          <td> {{ s.mp_starcat_time }} </td>
          <td> {{ "%.1f" | format(s.row) }} </td>
          <td> {{ "%.1f" | format(s.col) }} </td>
          <!-- td> {{ s.obs_ok }} </td -->
          <td> {{ s.n }} </td>
          <td> {{ s.n_ok }} </td>
          <td> {{ s.outliers }} </td>
          <td> {{ "%.1f" | format(100*s.f_track) }}% </td>
          <td> {{ "%.1f" | format(100*s.f_dr3) }}% </td>
          <td> {{ "%.1f" | format(100*s.f_dr5) }}% </td>
          <td> {{ "%.1f" | format(100*s.f_ok) }}% </td>
          <td> {{ "%.2f" | format(s.lf_variability_100s) }} </td>
          <td> {{ "%.2f" | format(s.t_mean) }} </td>
          <td> {{ "%.2f" | format(s.t_mean_err) }} </td>
          <td> {{ s.comments }} </td>
        </tr>
        {%- endfor %}
      </table>

    </div>
  </body>
</html>
"""


BAD_OBS_EMAIL_REPORT = """Warning in magnitude estimates at {{ date }}.

There were {{ bad_obs | length }} suspicious observation{% if bad_obs |length != 1 %}s{% endif %}
in magnitude estimates:
{% for s in bad_obs %}
- {{ "% 6d" | format(s.obsid) }}: time={{ s.mp_starcat_time }}, n={{ s.n }}, n_ok={{ s.n_ok }}, outliers={{ s.outliers }}, f_track={{ "%.1f" | format(100*s.f_track) }}%
{% endfor %}
"""


def email_bad_obs_report(bad_obs, to,
                         sender=f"{getpass.getuser()}@{platform.uname()[1]}"):
    date = CxoTime().date[:14]
    message = jinja2.Template(BAD_OBS_EMAIL_REPORT,
                              trim_blocks=True,
                              lstrip_blocks=True).render(bad_obs=bad_obs, date=date)
    msg = MIMEText(message)
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = \
        f"{len(bad_obs)} suspicious observation{'s' if len(bad_obs) else ''}" \
        f" in magnitude estimates at {date}."
    p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
    p.communicate(msg.as_string().encode())