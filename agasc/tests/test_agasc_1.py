# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile

import numpy as np
import pytest
import tables
from astropy.table import Table
from ska_helpers.utils import temp_env_var

import agasc
from agasc.agasc import _read_h5_table_cached, update_color1_column


def test_multi_agasc():
    tempdir = tempfile.mkdtemp()

    # Make two custom agasc files from the miniagasc, using 20 stars from
    # around the middle of the table
    with tables.open_file(agasc.default_agasc_file()) as h5:
        middle = int(len(h5.root.data) // 2)
        stars1 = Table(h5.root.data[middle : middle + 20])
        agasc.write_agasc(
            os.path.join(tempdir, "stars1.h5"),
            stars=stars1.as_array(),
            version="test",
            order=agasc.TableOrder.DEC,
            full_agasc=False,
        )
        # stars1.write(os.path.join(tempdir, "stars1.h5"), path="data")
        stars2 = Table(h5.root.data[middle + 20 : middle + 60])
        agasc.write_agasc(
            os.path.join(tempdir, "stars2.h5"),
            stars=stars2.as_array(),
            version="test",
            order=agasc.TableOrder.DEC,
            full_agasc=False,
        )
        # stars2.write(os.path.join(tempdir, "stars2.h5"), path="data")

    # Fetch all the stars from a custom agasc and make sure we have the right number of stars
    # with no errors
    all_stars2 = agasc.get_agasc_cone(
        0, 90, radius=180, agasc_file=os.path.join(tempdir, "stars2.h5")
    )
    assert len(all_stars2) == len(stars2)
    # Fetch all the stars from the other custom agasc and do the same.  The point of the two files
    # is to confirm that the caching behavior in agasc doesn't cause problems with fetches
    all_stars1 = agasc.get_agasc_cone(
        0, 90, radius=180, agasc_file=os.path.join(tempdir, "stars1.h5")
    )
    assert len(all_stars1) == len(stars1)

    # Do a position filtered search using the first star in the table as a reference and make sure
    # we get the same star from the reference agasc.  Do this with the stars2 file as this confirms
    # that we can switch back and forth between files and get the correct content.
    cone2 = agasc.get_agasc_cone(
        all_stars2["RA"][0],
        all_stars2["DEC"][0],
        radius=0.000001,
        agasc_file=os.path.join(tempdir, "stars2.h5"),
    )
    # And this is a read of the default agasc file after the custom ones so should confirm that
    # the custom files didn't break that access.
    cone2_full = agasc.get_agasc_cone(
        all_stars2["RA"][0], all_stars2["DEC"][0], radius=0.000001
    )
    assert cone2[0]["AGASC_ID"] == cone2_full[0]["AGASC_ID"]
    # Confirm that there is just one star in this test setup (not a module test, but confirms test
    # setup is as intended).
    assert len(cone2_full) == 1
    assert len(cone2) == len(cone2_full)


def test_update_color1_func():
    """
    Test code to update the COLOR1 column.
    """
    color1 = [1.0, 1.0, 1.5, 1.5, 1.5, 1.5]
    color2 = np.array([1.0, 1.0, 1.5, 1.5, 1.75, 2.0]) / 0.850
    rsv3 = [0, 1, 0, 1, 0, 1]
    # First two are left unchanged because color1 < 1.5
    # Third is still 1.5 because RSV3=0 (no good mag available so still "bad mag")
    # Fourth now gets COLOR1 = 1.499 because RSV3=1 => good mag
    # Fifth is still 1.5 because RSV3=0 (no good mag available so still "bad mag")
    # Sixth now gets COLOR1 = COLOR2 * 0.850 = 2.0
    stars = Table([color1, color2, rsv3], names=["COLOR1", "COLOR2", "RSV3"])
    update_color1_column(stars)

    assert np.allclose(stars["COLOR1"], [1.0, 1.0, 1.5, 1.499, 1.5, 2.0])
    assert np.allclose(stars["COLOR2"], color2)


def test_update_color1_get_star(proseco_agasc_1p7):
    """
    Test updated color1 in get_star() call.

     AGASC_ID  COLOR1  COLOR2 C1_CATID  RSV3
      int32   float32 float32  uint8   uint8
    --------- ------- ------- -------- -----
    981997696     1.5   1.765        5     0
    759439648     1.5   1.765        5     1

    """
    star = agasc.get_star(981997696)
    assert np.isclose(star["COLOR1"], 1.5)

    star = agasc.get_star(759439648)
    assert np.isclose(star["COLOR1"], 1.499)

    star = agasc.get_star(981997696, fix_color1=False)
    assert np.isclose(star["COLOR1"], 1.5)

    star = agasc.get_star(759439648, fix_color1=False)
    assert np.isclose(star["COLOR1"], 1.5)


def test_update_color1_get_agasc_cone(proseco_agasc_1p7):
    """
    Test updated color1 in get_agasc_cone() call.
    """
    ra, dec = 323.22831196, -13.11621348
    stars = agasc.get_agasc_cone(ra, dec, 0.2)
    stars.add_index("AGASC_ID")
    assert np.isclose(stars.loc[759960152]["COLOR1"], 1.60055, rtol=0, atol=0.0005)
    assert np.isclose(stars.loc[759439648]["COLOR1"], 1.499, rtol=0, atol=0.0005)

    stars = agasc.get_agasc_cone(ra, dec, 0.2, fix_color1=False)
    stars.add_index("AGASC_ID")
    assert np.isclose(stars.loc[759960152]["COLOR1"], 1.5, rtol=0, atol=0.0005)
    assert np.isclose(stars.loc[759439648]["COLOR1"], 1.5, rtol=0, atol=0.0005)


def test_get_agasc_filename(tmp_path, monkeypatch):
    monkeypatch.delenv("AGASC_HDF5_FILE", raising=False)
    monkeypatch.setenv("AGASC_DIR", str(tmp_path))
    names = [
        "agasc1p6.h5",
        "agasc1p7.h5",
        "agasc1p8.h5",
        "agasc1p8.hdf5",
        "agasc1p8rc2.h5",
        "proseco_agasc_1p6.h5",
        "proseco_agasc_1p7.h5",
        "proseco_agasc_1p8.h5",
        "proseco_agasc_1p8rc2.h5",
        "proseco_agasc_1p9rc1.h5",
        "miniagasc_1p6.h5",
        "miniagasc_1p7.h5",
        "miniagasC_1p7.h5",
        "miniagasc_1p10.h5",
        "miniagasc_2p8.h5",
    ]
    for name in names:
        (tmp_path / name).touch()

    def _check(filename, expected, allow_rc=False, version=None):
        assert agasc.get_agasc_filename(filename, allow_rc, version) == str(expected)

    # Default is latest proseco_agasc in AGASC_DIR
    _check(None, tmp_path / "proseco_agasc_1p8.h5")

    # Default is latest proseco_agasc in AGASC_DIR
    _check(None, tmp_path / "proseco_agasc_1p9rc1.h5", allow_rc=True)

    # With no wildcard just add .h5. File existence is not required by this function.
    with pytest.raises(ValueError, match=r"agasc_file must end with '\*' or '.h5'"):
        _check("agasc1p6", tmp_path / "agasc1p6.h5")

    # Doesn't find the rc2 version regardless of allow_rc (agasc_1p8.h5 wins over
    # agasc_1p8rc2.h5).
    _check("agasc*", tmp_path / "agasc1p8.h5", allow_rc=False)
    _check("agasc*", tmp_path / "agasc1p8.h5", allow_rc=True)

    # 1p8rc2 is available but it takes the non-RC version 1p8
    _check(
        "proseco_agasc_*",
        tmp_path / "proseco_agasc_1p8.h5",
        allow_rc=True,
        version="1p8",
    )
    # You can choose the RC version explicitly
    _check(
        "proseco_agasc_*",
        tmp_path / "proseco_agasc_1p8rc2.h5",
        allow_rc=True,
        version="1p8rc2",
    )
    # For version="1p9" only the 1p9rc1 version is available
    _check(
        "proseco_agasc_*",
        tmp_path / "proseco_agasc_1p9rc1.h5",
        allow_rc=True,
        version="1p9",
    )

    # Wildcard finds the right file (and double-digit version is OK)
    _check("miniagasc_*", tmp_path / "miniagasc_1p10.h5")

    # With .h5 supplied just return the file, again don't require existence.
    _check("agasc1p7.h5", "agasc1p7.h5")
    _check("doesnt-exist.h5", "doesnt-exist.h5")

    # With AGASC_HDF5_FILE set, use that if agasc_file is None
    with temp_env_var("AGASC_HDF5_FILE", "proseco_agasc_1p5.h5"):
        _check(None, tmp_path / "proseco_agasc_1p5.h5")
        # Explicit agasc_file overrides AGASC_HDF5_FILE
        _check("agasc1p7.h5", "agasc1p7.h5")

    # With a glob pattern existence of a matching file is required
    with pytest.raises(FileNotFoundError, match="No AGASC files"):
        agasc.get_agasc_filename("doesnt-exist*")


@pytest.mark.parametrize("agasc_version", ["1p7", "1p8"])
def test_cache_and_columns(agasc_version):
    """Ensure that caching doesn't affect the outputs of get_agasc_cone().

    Also make sure that setting `columns` has the desired result.
    """
    agasc_file = agasc.get_agasc_filename(version=agasc_version, allow_rc=True)
    n_iter = 10
    np.random.seed(0)
    ras = np.random.uniform(0, 360, n_iter)
    decs = np.random.uniform(-90, 90, n_iter)

    # fmt: off
    colnames = (
        "AGASC_ID", "RA", "DEC", "PM_RA", "PM_DEC", "EPOCH", "MAG_ACA",
        "RA_PMCORR", "DEC_PMCORR",
    )
    colnames_all = [
        "AGASC_ID", "RA", "DEC", "POS_ERR", "EPOCH", "PM_RA", "PM_DEC",
        "MAG_ACA", "MAG_ACA_ERR", "CLASS", "COLOR1", "COLOR2", "RSV1", "RSV3", "VAR",
        "ASPQ1", "ASPQ2", "ASPQ3", "RA_PMCORR", "DEC_PMCORR"
    ]
    # fmt: on

    _read_h5_table_cached.cache_clear()
    assert _read_h5_table_cached.cache_info().hits == 0
    assert _read_h5_table_cached.cache_info().currsize == 0

    # Prime the cache with 2 entries (one for each values of columns)
    for columns in (colnames, None):
        agasc.get_agasc_cone(
            cache=True, columns=columns, ra=0, dec=0, agasc_file=agasc_file
        )
    assert _read_h5_table_cached.cache_info().currsize == 2
    assert _read_h5_table_cached.cache_info().misses == 2

    stars = {}
    for ra, dec in zip(ras, decs):
        for cache in (False, True):
            for columns in (None, colnames):
                stars[cache, columns] = agasc.get_agasc_cone(
                    cache=cache,
                    columns=columns,
                    ra=ra,
                    dec=dec,
                    agasc_file=agasc_file,
                    date="2019:001",
                )

            assert stars[cache, None].colnames == colnames_all
            assert stars[cache, colnames].colnames == list(colnames)

        assert np.all(stars[True, None] == stars[False, None])
        assert np.all(stars[True, colnames] == stars[False, colnames])

    # Still just two cache entries
    assert _read_h5_table_cached.cache_info().misses == 2
    assert _read_h5_table_cached.cache_info().currsize == 2
