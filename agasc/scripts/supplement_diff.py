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

import agasc


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


def table_diff(file_from, file_to):
    table_names = ["from", "to"]
    agasc_filename = agasc.get_agasc_filename()
    with tables.open_file(agasc_filename) as h5:
        stars = table.Table(h5.root.data[:])
    stars.rename_columns(["AGASC_ID", "MAG_ACA"], ["agasc_id", "mag_aca"])

    with tables.open_file(file_from) as h5_from, tables.open_file(file_to) as h5_to:
        mags = table.join(
            table.Table(h5_from.get_node("/mags")[:]),
            table.Table(h5_to.get_node("/mags")[:]),
            join_type="outer",
            table_names=table_names,
            keys="agasc_id",
            metadata_conflicts="silent",
        )
        # the `bad` table will be joined with the `mags` table.
        # I rename the columns so the names are self-explanatory.
        b1 = table.Table(h5_from.get_node("/bad")[:])
        b1.rename_column("source", "bad_star_source")
        b2 = table.Table(h5_to.get_node("/bad")[:])
        b2.rename_column("source", "bad_star_source")
        bad = table.join(
            b1,
            b2,
            join_type="outer",
            table_names=table_names,
            keys="agasc_id",
            metadata_conflicts="silent",
        )

    # missing magnitudes in the supplement are filled with catalog magnitudes
    mags = table.join(
        mags,
        stars[["agasc_id", "mag_aca"]],
        keys="agasc_id",
        join_type="left",
        metadata_conflicts="silent",
    )
    for name in table_names:
        if hasattr(mags[f"mag_aca_{name}"], "mask"):
            mags[f"mag_aca_{name}"][mags[f"mag_aca_{name}"].mask] = mags["mag_aca"][
                mags[f"mag_aca_{name}"].mask
            ]

    mags = table.join(mags, bad, join_type="outer")

    # filling remaining masked values so they can be compared
    for name in table_names:
        mags[f"bad_star_source_{name}"][mags[f"bad_star_source_{name}"].mask] = -9999
        mags[f"mag_aca_{name}"][mags[f"mag_aca_{name}"].mask] = -9999

    # now we select the rows that changed
    mag_changed = (
        mags[f"mag_aca_{table_names[0]}"] != mags[f"mag_aca_{table_names[1]}"]
    ) | (
        mags[f"bad_star_source_{table_names[0]}"]
        != mags[f"bad_star_source_{table_names[1]}"]
    )
    for name in table_names:
        if hasattr(mags[f"mag_aca_{name}"], "mask"):
            mag_changed |= mags[f"mag_aca_{name}"].mask

    mags = mags[mag_changed]

    mags["d_mag"] = (
        mags[f"mag_aca_{table_names[1]}"] - mags[f"mag_aca_{table_names[0]}"]
    )
    mags["d_mag_err"] = (
        mags[f"mag_aca_err_{table_names[1]}"] - mags[f"mag_aca_err_{table_names[0]}"]
    )

    return mags


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

    if args.o.suffix == ".html":
        with open(args.o, "w") as fh:
            diff = diff_to_html(
                diff_files(args.fromfile, args.tofile, exclude_mags=args.exclude_mags)
            )
            fh.write(diff)
    else:
        mags = table_diff(args.fromfile, args.tofile)
        mags.write(args.o, overwrite=True)


if __name__ == "__main__":
    main()
