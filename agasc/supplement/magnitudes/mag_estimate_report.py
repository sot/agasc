import platform
import getpass
from subprocess import Popen, PIPE
from pathlib import Path
from email.mime.text import MIMEText
import jinja2
import numpy as np
import matplotlib.pyplot as plt
from astropy import table
from cxotime import CxoTime

from agasc.supplement.magnitudes import mag_estimate


JINJA2 = jinja2.Environment(
    loader=jinja2.PackageLoader('agasc.supplement.magnitudes', 'templates'),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)


class MagEstimateReport:
    def __init__(self, agasc_stats, obs_stats, directory='./mag_estimates_reports'):
        self.agasc_stats = agasc_stats
        self.obs_stats = obs_stats
        self.directory = Path(directory)

    def single_star_html(self, agasc_id, directory,
                         static_dir='https://cxc.cfa.harvard.edu/mta/ASPECT/www_resources',
                         highlight_obs=lambda o: ~o['obs_ok']):
        if np.sum(self.agasc_stats['agasc_id'] == agasc_id) == 0:
            return

        star_template = JINJA2.get_template('star_report.html')

        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)

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
        fig = self.plot_set(agasc_id, args=args, filename=directory / 'mag_stats.png')
        plt.close(fig)

        with open(directory / 'index.html', 'w') as out:
            out.write(star_template.render(agasc_stats=s,
                                           obs_stats=o.as_array(),
                                           static_dir=static_dir))
        return directory / 'index.html'

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
        run_template = JINJA2.get_template('run_report.html')

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
                'stars': self.agasc_stats['agasc_id'][
                    ~np.in1d(self.agasc_stats['agasc_id'], agasc_ids)]
            })
            agasc_ids = self.agasc_stats['agasc_id']
        failed_agasc_ids = [f['agasc_id'] for f in fails
                            if f['agasc_id'] and int(f['agasc_id']) in self.obs_stats['agasc_id']]
        agasc_ids = np.unique(np.concatenate([agasc_ids, failed_agasc_ids]))

        # this turns all None into '' in a new list of failures
        fails = [{k: '' if v is None else v for k, v in f.items()} for i, f in enumerate(fails)]

        agasc_stats = self.agasc_stats.copy()

        # check how many observations were added in this run, and how many of those are ok
        new_obs_mask = ((self.obs_stats['mp_starcat_time'] >= info["tstart"])
                        & (self.obs_stats['mp_starcat_time'] <= info["tstop"]))
        if np.any(new_obs_mask):
            new_obs = self.obs_stats[new_obs_mask]. \
                group_by('agasc_id')[['agasc_id', 'obsid', 'obs_ok']]. \
                groups.aggregate(np.count_nonzero)[['agasc_id', 'obsid', 'obs_ok']]
            new_obs['n_obs_bad_new'] = new_obs['obsid'] - new_obs['obs_ok']

            all_agasc_ids = np.unique(np.concatenate([
                new_obs['agasc_id'],
                [f['agasc_id'] for f in fails]
            ]))
            agasc_stats = table.join(agasc_stats, new_obs[['agasc_id', 'n_obs_bad_new']],
                                     keys=['agasc_id'])

            assert np.all(np.in1d(agasc_stats['agasc_id'], all_agasc_ids)
                          ), 'Not all AGASC IDs are in new obs.'

        # add some extra fields
        if len(agasc_stats):
            if 'n_obs_bad_new' not in agasc_stats.colnames:
                agasc_stats['n_obs_bad_new'] = 0
            agasc_stats['n_obs_bad'] = agasc_stats['n_obsids'] - agasc_stats['n_obsids_ok']
            agasc_stats['flag'] = '          '
            agasc_stats['flag'][:] = ''
            agasc_stats['flag'][agasc_stats['n_obs_bad'] > 0] = 'warning'
            agasc_stats['flag'][agasc_stats['n_obs_bad_new'] > 0] = 'danger'
            agasc_stats['delta'] = (agasc_stats['t_mean_dr3'] - agasc_stats['mag_aca'])
            agasc_stats['sigma'] = ((agasc_stats['t_mean_dr3'] - agasc_stats['mag_aca'])
                                    / agasc_stats['mag_aca_err'])
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

        tooltips = {
            'warning': 'At least one bad observation',
            'danger': 'At least one new bad observation'
        }

        # make all individual star reports
        star_reports = {}
        for agasc_id in np.atleast_1d(agasc_ids):
            try:
                dirname = self.directory / 'stars' / f'{agasc_id//1e7:03.0f}' / f'{agasc_id:.0f}'
                if make_single_reports:
                    self.single_star_html(
                        agasc_id,
                        directory=dirname,
                        highlight_obs=highlight_obs
                    )
                if dirname.exists():
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
        star_reports = {i: str(star_reports[i].relative_to(self.directory)) for i in star_reports}
        # make report
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        with open(self.directory / filename, 'w') as out:
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
                if (np.isfinite(obs_stats[sel][0]['t_mean'])
                        and np.isfinite(obs_stats[sel][0]['t_std'])):
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

        if (draw_agasc_mag_stats
                and np.isfinite(agasc_stat['mag_obs']) and agasc_stat['mag_obs'] > 0):
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

        all_ok = ((timeline['AOACASEQ'] == 'KALM')
                  & (timeline['AOPCADMD'] == 'NPNT')
                  & (timeline['AOACFCT'] == 'TRAK')
                  & (timeline['AOACIIR'] == 'OK')
                  & (timeline['AOACISP'] == 'OK')
                  & (timeline['dr'] < 3)
                  )
        flags = [
            ('dr > 5', ((timeline['AOACASEQ'] == 'KALM')
                        & (timeline['AOPCADMD'] == 'NPNT')
                        & (timeline['AOACFCT'] == 'TRAK')
                        & (timeline['dr'] >= 5))),
            ('Ion. rad.', (timeline['AOACIIR'] != 'OK')),
            ('Sat. pixel.', (timeline['AOACISP'] != 'OK')),
            ('not track', ((timeline['AOACASEQ'] == 'KALM')
                           & (timeline['AOPCADMD'] == 'NPNT')
                           & (timeline['AOACFCT'] != 'TRAK'))),
            ('not Kalman', ((timeline['AOACASEQ'] != 'KALM')
                            | (timeline['AOPCADMD'] != 'NPNT'))),
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


def email_bad_obs_report(bad_obs, to,
                         sender=f"{getpass.getuser()}@{platform.uname()[1]}"):
    date = CxoTime().date[:14]
    message = JINJA2.get_template('email_report.txt')

    msg = MIMEText(message)
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = \
        f"{len(bad_obs)} suspicious observation{'s' if len(bad_obs) else ''}" \
        f" in magnitude estimates at {date}."
    p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
    p.communicate(msg.as_string().encode())
