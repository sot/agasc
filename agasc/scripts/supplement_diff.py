#!/usr/bin/env python

"""
Generate diff between to supplement files and output in HTML format.
"""

import tables
from pathlib import Path
import difflib
import pygments.lexers
import pygments.formatters
import argparse
import datetime

from io import StringIO

from astropy.io import ascii
from astropy import table


def read_file(filename, exclude=[]):
    formats = {
        'agasc_versions': {k: '{:>21s}' for k in ['mags', 'obs', 'bad', 'supplement']},
        'last_updated': {k: '{:>21s}' for k in ['mags', 'obs', 'bad', 'supplement']},
        'obs': {
            'comments': '{:>80s}',
            'agasc_id': '{:10d}'
        },
    }

    node_names = ['agasc_versions', 'last_updated', 'obs', 'bad']
    with tables.open_file(filename) as h5:
        for node in h5.root:
            if node.name not in node_names:
                node_names.append(node.name)
        all_lines = []
        for name in node_names:
            if name in exclude:
                continue
            node = h5.get_node(f'/{name}')
            t = table.Table(node[:])
            t.convert_bytestring_to_unicode
            s = StringIO()
            ascii.write(t, s, format='fixed_width', formats=formats.get(name, {}))
            s.seek(0)
            lines = s.readlines()
            dashes = "-" * (len(lines[0]) - 1) + "\n"
            all_lines += ([dashes, f'| {node.name}\n', dashes] + lines + [dashes])
    return all_lines


def diff_files(fromfile, tofile, include_mags=False):
    exclude = [] if include_mags else ['mags']
    fromlines = read_file(fromfile, exclude)
    tolines = read_file(tofile, exclude)
    diff = difflib.unified_diff(fromlines, tolines, str(fromfile), str(tofile))
    diff = ''.join(diff)
    return diff


def diff_to_html(diff):
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lexer = pygments.lexers.DiffLexer()
    formatter = pygments.formatters.HtmlFormatter(
        full=True, linenos='table',
        title=f'AGASC supplement diff - {date}')
    return pygments.highlight(diff, lexer, formatter)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fromfile', type=Path, help='The original supplement file.')
    parser.add_argument('tofile', type=Path, help='The modified supplement file.')
    parser.add_argument(
        '-o',
        help='Output HTML file',
        type=Path,
        default='agasc_supplement_diff.html'
    )
    parser.add_argument(
        '--include-mags', default=False, action='store_true',
        help='Include changes in the "mags" table. There can be many of these.',
    )
    return parser


def main():
    args = get_parser().parse_args()
    assert args.fromfile.exists()
    assert args.tofile.exists()

    if not args.o.parent.exists():
        args.o.parent.mkdir()

    with open(args.o, 'w') as fh:
        diff = diff_to_html(diff_files(args.fromfile, args.tofile, include_mags=args.include_mags))
        fh.write(diff)


if __name__ == "__main__":
    main()
