#!/usr/bin/env python

"""
Generate diff between to supplement files and output in HTML format.
"""

import argparse
import datetime
import difflib
import os
from io import StringIO
from pathlib import Path

import pygments.formatters
import pygments.lexers
import tables
from astropy import table
from astropy.io import ascii


def read_file(filename, exclude=None):
    if exclude is None:
        exclude = []
    formats = {
        "agasc_versions": {k: "{:>21s}" for k in ["mags", "obs", "bad", "supplement"]},
        "last_updated": {k: "{:>21s}" for k in ["mags", "obs", "bad", "supplement"]},
        "obs": {"comments": "{:>80s}", "agasc_id": "{:10d}"},
    }

    node_names = ["agasc_versions", "last_updated", "obs", "bad"]
    with tables.open_file(filename) as h5:
        for node in h5.root:
            if node.name not in node_names:
                node_names.append(node.name)
        all_lines = []
        for name in node_names:
            if name in exclude:
                continue
            node = h5.get_node(f"/{name}")
            t = table.Table(node[:])
            t.convert_bytestring_to_unicode()
            s = StringIO()
            ascii.write(t, s, format="fixed_width", formats=formats.get(name, {}))
            s.seek(0)
            lines = s.readlines()
            dashes = "-" * (len(lines[0]) - 1) + "\n"
            all_lines += [dashes, f"| {node.name}\n", dashes] + lines + [dashes]
    return all_lines


def diff_files(fromfile, tofile, exclude_mags=False):
    exclude = ["mags"] if exclude_mags else []
    fromlines = read_file(fromfile, exclude)
    tolines = read_file(tofile, exclude)
    diff = difflib.unified_diff(fromlines, tolines, str(fromfile), str(tofile))
    diff = "".join(diff)
    return diff


def diff_to_html(diff):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lexer = pygments.lexers.DiffLexer()
    formatter = pygments.formatters.HtmlFormatter(
        full=True, linenos="table", title=f"AGASC supplement diff - {date}"
    )
    return pygments.highlight(diff, lexer, formatter)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="fromfile", type=Path, help="The original supplement file."
    )
    parser.add_argument(
        "--to", dest="tofile", type=Path, help="The modified supplement file."
    )
    parser.add_argument(
        "-o", help="Output HTML file", type=Path, default="agasc_supplement_diff.html"
    )
    parser.add_argument(
        "--exclude-mags",
        default=False,
        action="store_true",
        help='Exclude changes in the "mags" table. There can be many of these.',
    )
    return parser


def main():
    args = get_parser().parse_args()
    if args.fromfile is None and "SKA" in os.environ:
        args.fromfile = (
            Path(os.environ["SKA"]) / "data" / "agasc" / "agasc_supplement.h5"
        )
    if args.tofile is None and "SKA" in os.environ:
        args.tofile = (
            Path(os.environ["SKA"]) / "data" / "agasc" / "rc" / "agasc_supplement.h5"
        )
    assert args.tofile, 'Option "--to" was not given and SKA is not defined'
    assert args.fromfile, 'Option "--from" was not given and SKA is not defined'
    assert args.fromfile.exists(), f"File {args.fromfile} does not exist"
    assert args.tofile.exists(), f"File {args.tofile} does not exist"

    if not args.o.parent.exists():
        args.o.parent.mkdir()

    with open(args.o, "w") as fh:
        diff = diff_to_html(
            diff_files(args.fromfile, args.tofile, exclude_mags=args.exclude_mags)
        )
        fh.write(diff)


if __name__ == "__main__":
    main()
