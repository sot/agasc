import builtins
import io
import os
import pathlib

import numpy as np
import pytest
import tables
from astropy import table

from agasc.scripts import update_supplement
from agasc.supplement.magnitudes import mag_estimate, star_obs_catalogs
from agasc.supplement.utils import BAD_DTYPE, OBS_DTYPE

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"

# for the purposes of the supplement tables
# star_obs_catalogs.STARS_OBS is used to determine all AGASC IDs in an observation
# or the last observation time of a given AGASC_ID.
# this array is used to monkey-patch star_obs_catalogs.STARS_OBS
STARS_OBS = np.array(
    [
        (56314, 114950168, "2010:110:14:57:43.442"),
        (56314, 114950584, "2010:110:14:57:43.442"),
        (56314, 114952056, "2010:110:14:57:43.442"),
        (56314, 114952792, "2010:110:14:57:43.442"),
        (56314, 114952824, "2010:110:14:57:43.442"),
        (56314, 114955056, "2010:110:14:57:43.442"),
        (56314, 114956608, "2010:110:14:57:43.442"),
        (56314, 115347520, "2010:110:14:57:43.442"),
        (56312, 357045496, "2010:110:16:59:51.399"),
        (56312, 357049064, "2010:110:16:59:51.399"),
        (56312, 357051640, "2010:110:16:59:51.399"),
        (56312, 357054680, "2010:110:16:59:51.399"),
        (56312, 358220224, "2010:110:16:59:51.399"),
        (56312, 358222256, "2010:110:16:59:51.399"),
        (56312, 358224400, "2010:110:16:59:51.399"),
        (56312, 358757768, "2010:110:16:59:51.399"),
        (56313, 441853632, "2010:110:15:07:49.876"),
        (56313, 441854760, "2010:110:15:07:49.876"),
        (56313, 441855776, "2010:110:15:07:49.876"),
        (56313, 441856032, "2010:110:15:07:49.876"),
        (56313, 441856400, "2010:110:15:07:49.876"),
        (56313, 441980072, "2010:110:15:07:49.876"),
        (56313, 491391592, "2010:110:15:07:49.876"),
        (56313, 491394504, "2010:110:15:07:49.876"),
        (56311, 563087864, "2010:110:18:59:46.600"),
        (56311, 563088952, "2010:110:18:59:46.600"),
        (56311, 563089432, "2010:110:18:59:46.600"),
        (56311, 563091784, "2010:110:18:59:46.600"),
        (56311, 563092520, "2010:110:18:59:46.600"),
        (56311, 563612488, "2010:110:18:59:46.600"),
        (56311, 563612792, "2010:110:18:59:46.600"),
        (56311, 563617352, "2010:110:18:59:46.600"),
        (56310, 624826320, "2010:110:20:33:54.789"),
        (56310, 624826464, "2010:110:20:33:54.789"),
        (56310, 624828488, "2010:110:20:33:54.789"),
        (56310, 624831328, "2010:110:20:33:54.789"),
        (56310, 624831392, "2010:110:20:33:54.789"),
        (56310, 624954248, "2010:110:20:33:54.789"),
        (56310, 624956216, "2010:110:20:33:54.789"),
        (56310, 625476960, "2010:110:20:33:54.789"),
        (12203, 697581832, "2010:111:10:30:46.876"),
        (12203, 697963056, "2010:111:10:30:46.876"),
        (12203, 697963288, "2010:111:10:30:46.876"),
        (12203, 697970576, "2010:111:10:30:46.876"),
        (12203, 697973824, "2010:111:10:30:46.876"),
        (56308, 732697144, "2010:110:23:23:49.708"),
        (56308, 732698416, "2010:110:23:23:49.708"),
        (56309, 762184312, "2010:110:22:02:30.780"),
        (56309, 762184768, "2010:110:22:02:30.780"),
        (56309, 762185584, "2010:110:22:02:30.780"),
        (56309, 762186016, "2010:110:22:02:30.780"),
        (56309, 762186080, "2010:110:22:02:30.780"),
        (56309, 762191224, "2010:110:22:02:30.780"),
        (56309, 762579584, "2010:110:22:02:30.780"),
        (56309, 762581024, "2010:110:22:02:30.780"),
        (56308, 806748432, "2010:110:23:23:49.708"),
        (56308, 806748880, "2010:110:23:23:49.708"),
        (56308, 806750112, "2010:110:23:23:49.708"),
        (56308, 806750408, "2010:110:23:23:49.708"),
        (56308, 806750912, "2010:110:23:23:49.708"),
        (56308, 806751424, "2010:110:23:23:49.708"),
        (56306, 956708808, "2010:111:02:18:49.052"),
        (56306, 957219128, "2010:111:02:18:49.052"),
        (56306, 957221200, "2010:111:02:18:49.052"),
        (56306, 957222432, "2010:111:02:18:49.052"),
        (56306, 957229080, "2010:111:02:18:49.052"),
        (56306, 957230976, "2010:111:02:18:49.052"),
        (56306, 957233920, "2010:111:02:18:49.052"),
        (56306, 957369088, "2010:111:02:18:49.052"),
        (11849, 1019347720, "2010:111:12:26:54.536"),
        (11849, 1019348536, "2010:111:12:26:54.536"),
        (11849, 1019350904, "2010:111:12:26:54.536"),
        (11849, 1019354232, "2010:111:12:26:54.536"),
        (11849, 1019357032, "2010:111:12:26:54.536"),
        (11980, 1198184872, "2010:111:03:21:11.299"),
        (11980, 1198190648, "2010:111:03:21:11.299"),
        (11980, 1198190664, "2010:111:03:21:11.299"),
        (11980, 1198191400, "2010:111:03:21:11.299"),
        (11980, 1198192456, "2010:111:03:21:11.299"),
    ],
    dtype=[("obsid", "<i8"), ("agasc_id", "<i8"), ("mp_starcat_time", "<U21")],
)


TEST_YAML = {
    # file_0.yml is what is used to create the base test agasc_supplement.h5
    "file_0.yml": """
                  obs:
                    - obsid: 56311
                      status: 1
                    - obsid: 56308
                      status: 0
                      agasc_id: [806750112]
                    - obsid: 11849
                      status: 1
                      agasc_id: [1019348536, 1019350904]
                      comments: just removed them
                  bad:
                    77073552: 2
                  mags:
                    - agasc_id: 5
                      mag_aca: 5.5
                      mag_aca_err: 0.3
                      last_obs_time: 2020:001
                    - agasc_id: 114950168
                      mag_aca: 7.
                      mag_aca_err: 0.2
                      last_obs_time: 2010:110:14:57:43.442
                  """,
    "file_4.yml": """
                  obs:
                    - obsid: 56314
                      status: 1
                      comments: removed because I felt like it
                      agasc_id: [114950168, 114950584, 114952056, 114952792,
                                 114952824, 114955056, 114956608, 115347520]
                  bad:
                    77073552: null
                    23434: 10
                  """,
    "file_5.yml": """
                  obs:
                    - obsid: 56314
                      status: 1
                      comments: removed because I felt like it
                  bad:
                    77073552: null
                    23434: 10
                  """,
    "file_6.yml": """
                  obs:
                    - obsid: 56314
                      status: 1
                      agasc_id: 114950168
                  """,
    "file_7.yml": """
                  obs:
                    - obsid: 56311
                      status: 1
                    - obsid: 56308
                      status: 0
                      agasc_id: [806750112]
                    - obsid: 11849
                      status: 1
                      agasc_id: [1019348536, 1019350904]
                      comments: just removed them
    """,
    "file_1.yml": """
                  bad:
                    77073552: null
                    23434: 10
                  """,
    "file_2.yml": """
                  bad:
                    77073552: null
                    23434: 10
                  """,
    "file_8.yml": """
              mags:
                - agasc_id: 806748880
                  mag_aca: 12.
                  mag_aca_err: 1.
              """,
    "file_9.yml": """
              bad:
                77073552: 11
                77073552: 10
                """,
    "file_10.yml": """
              mags:
                - agasc_id: 806748880
                  mag_aca: 12.
                  mag_aca_err: 1.
                  last_obs_time: 2020:001:00:00
                """,
}


TEST_DATA = {
    "file_0.yml": {
        "obs": [
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563087864,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563088952,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563089432,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563091784,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563092520,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563612488,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563612792,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563617352,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:23:23:49.708",
                "obsid": 56308,
                "agasc_id": 806750112,
                "status": 0,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:111:12:26:54.536",
                "obsid": 11849,
                "agasc_id": 1019348536,
                "status": 1,
                "comments": "just removed them",
            },
            {
                "mp_starcat_time": "2010:111:12:26:54.536",
                "obsid": 11849,
                "agasc_id": 1019350904,
                "status": 1,
                "comments": "just removed them",
            },
        ],
        "bad": [(77073552, 2)],
        "mags": [
            {
                "agasc_id": 5,
                "mag_aca": 5.5,
                "mag_aca_err": 0.3,
                "last_obs_time": 694224069.184,
            },
            {
                "agasc_id": 114950168,
                "mag_aca": 7.0,
                "mag_aca_err": 0.2,
                "last_obs_time": 388162729.626,
            },
        ],
    },
    "file_1.yml": {"obs": [], "bad": [(77073552, None), (23434, 10)], "mags": []},
    "file_2.yml": {"obs": [], "bad": [(77073552, None), (23434, 10)], "mags": []},
    "file_3.yml": {"obs": [], "bad": [], "mags": []},
    "file_4.yml": {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950168,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950584,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952056,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952792,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952824,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114955056,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114956608,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 115347520,
                "status": 1,
                "comments": "removed because I felt like it",
            },
        ],
        "bad": [(77073552, None), (23434, 10)],
        "mags": [],
    },
    "file_5.yml": {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950168,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950584,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952056,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952792,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952824,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114955056,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114956608,
                "status": 1,
                "comments": "removed because I felt like it",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 115347520,
                "status": 1,
                "comments": "removed because I felt like it",
            },
        ],
        "bad": [(77073552, None), (23434, 10)],
        "mags": [],
    },
    "file_6.yml": {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950168,
                "status": 1,
                "comments": "",
            }
        ],
        "bad": [],
        "mags": [],
    },
    "file_7.yml": {
        "obs": [
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563087864,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563088952,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563089432,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563091784,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563092520,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563612488,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563612792,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:18:59:46.600",
                "obsid": 56311,
                "agasc_id": 563617352,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:23:23:49.708",
                "obsid": 56308,
                "agasc_id": 806750112,
                "status": 0,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:111:12:26:54.536",
                "obsid": 11849,
                "agasc_id": 1019348536,
                "status": 1,
                "comments": "just removed them",
            },
            {
                "mp_starcat_time": "2010:111:12:26:54.536",
                "obsid": 11849,
                "agasc_id": 1019350904,
                "status": 1,
                "comments": "just removed them",
            },
        ],
        "bad": [],
        "mags": [],
    },
    "file_8.yml": {
        "obs": [],
        "bad": [],
        "mags": [
            {
                "agasc_id": 806748880,
                "mag_aca": 12.0,
                "mag_aca_err": 1.0,
                "last_obs_time": 388193095.892,
            }
        ],
    },
    "file_9.yml": {"bad": [(77073552, 10)], "obs": [], "mags": []},
    "file_10.yml": {
        "obs": [],
        "bad": [],
        "mags": [
            {
                "agasc_id": 806748880,
                "mag_aca": 12.0,
                "mag_aca_err": 1.0,
                "last_obs_time": 694224069.184,
            }
        ],
    },
}


def _open(filename, mode="r", **kwargs):
    if filename in TEST_YAML and mode == "r":
        return io.StringIO(TEST_YAML[filename])
    return builtins.default_open(filename, mode, **kwargs)


@pytest.fixture
def mock_open(monkeypatch):
    monkeypatch.setitem(__builtins__, "default_open", builtins.open)
    monkeypatch.setitem(__builtins__, "open", _open)


def test_parse_file(monkeypatch, mock_open):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    for filename in TEST_YAML:
        data = update_supplement._parse_obs_status_file(filename)
        print("ref:")
        print(TEST_DATA[filename])
        print("arg:")
        print(data)
        data = update_supplement._sanitize_args(data)
        assert data == TEST_DATA[filename], (
            f'_parse_obs_status_file("{filename}") == TEST_DATA["{filename}"]'
        )

        # _parse_obs_status_file should be idempotent
        data = update_supplement._sanitize_args(data)
        assert data == TEST_DATA[filename], (
            "update_supplement._sanitize_args should be idempotent"
        )


def test_parse_args_file(monkeypatch, mock_open):
    with pytest.raises(RuntimeError, match=r"catalog"):
        _ = update_supplement.parse_args(filename="file_4.yml")

    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    for filename in TEST_YAML:
        print(filename)
        ref = update_supplement.parse_args(filename=filename)
        print("ref")
        print(TEST_DATA[filename])
        print("data")
        print(ref)
        assert TEST_DATA[filename] == ref


def test_parse_args_bad(monkeypatch, mock_open):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    #######################
    # specifying bad stars
    #######################

    status = update_supplement.parse_args(bad_star_id=1, bad_star_source=2)
    assert status["obs"] == []
    assert status["bad"] == [(1, 2)]

    # bad star can be a list
    status = update_supplement.parse_args(bad_star_id=[1, 2], bad_star_source=3)
    assert status["obs"] == []
    assert status["bad"] == [(1, 3), (2, 3)]

    # if you specify bad_star, you must specify bad_star_source
    with pytest.raises(RuntimeError, match=r"specify bad_star_source"):
        update_supplement.parse_args(bad_star_id=1)


def test_parse_args_obs(monkeypatch, mock_open):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    #######################
    # specifying obs status
    #######################

    with pytest.raises(Exception, match=r"catalog has no observation"):
        status = update_supplement.parse_args(
            obsid=1, status=0, agasc_id=2, comments="comment"
        )

    status = update_supplement.parse_args(
        obsid=56314, status=1, comments="some comment"
    )
    ref = {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950168,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950584,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952056,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952792,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952824,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114955056,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114956608,
                "status": 1,
                "comments": "some comment",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 115347520,
                "status": 1,
                "comments": "some comment",
            },
        ],
        "bad": [],
        "mags": [],
    }
    assert status == ref

    # comments are optional
    status = update_supplement.parse_args(obsid=56314, status=1)
    ref = {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950168,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114950584,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952056,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952792,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114952824,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114955056,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 114956608,
                "status": 1,
                "comments": "",
            },
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 115347520,
                "status": 1,
                "comments": "",
            },
        ],
        "bad": [],
        "mags": [],
    }
    assert status == ref

    # optional agasc_id can be int or list
    status = update_supplement.parse_args(
        obsid=56314, status=0, agasc_id=[2], comments="comment"
    )
    ref = {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 2,
                "status": 0,
                "comments": "comment",
            }
        ],
        "bad": [],
        "mags": [],
    }
    assert status == ref

    status = update_supplement.parse_args(
        obsid=56314, status=0, agasc_id=2, comments="comment"
    )
    ref = {
        "obs": [
            {
                "mp_starcat_time": "2010:110:14:57:43.442",
                "obsid": 56314,
                "agasc_id": 2,
                "status": 0,
                "comments": "comment",
            }
        ],
        "bad": [],
        "mags": [],
    }
    assert status == ref


def test_parse_args(monkeypatch, mock_open):
    import copy

    # calling function before catalog is initialized gives an exception
    with pytest.raises(RuntimeError, match=r"catalog"):
        _ = update_supplement.parse_args("file_5.yml")

    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    filename = "file_4.yml"

    # can not specify bad star with different source in the file and in args.
    with pytest.raises(RuntimeError, match=r"name collision"):
        _ = update_supplement.parse_args(
            filename=filename, bad_star_id=23434, bad_star_source=12
        )

    # can specify bad star in the file and in args if the source is the same.
    status = update_supplement.parse_args(
        filename=filename, bad_star_id=23434, bad_star_source=10
    )
    ref = copy.deepcopy(TEST_DATA[filename])
    assert ref == status

    # If there are no name conflicts, args and file are merged
    # The following checks that parsing the file and command-line args at once is the same as
    # parsing the file, [arsing] the command-line args, concatenating and removing duplicates.
    status = update_supplement.parse_args(
        filename=filename,
        obs=56309,
        agasc_id=[762184312, 762184768, 762185584, 762186016],
        status=False,
        bad_star_id=[1, 2],
        bad_star_source=1000,
    )
    ref = update_supplement.parse_args(filename=filename)
    ref_2 = update_supplement.parse_args(
        obs=56309,
        agasc_id=[762184312, 762184768, 762185584, 762186016],
        status=False,
        bad_star_id=[1, 2],
        bad_star_source=1000,
    )

    ref["obs"] = table.unique(table.Table(ref["obs"] + ref_2["obs"]), keep="last")
    ref["bad"] = _remove_list_duplicates(ref["bad"] + ref_2["bad"])
    status["obs"] = table.Table(status["obs"])

    assert np.all(ref["obs"] == status["obs"])
    assert np.all(ref["mags"] == status["mags"])
    assert ref["bad"] == status["bad"]


def _disabled_write(*args, **kwargs):
    raise Exception("Tried to write file when it should not")


def test_update_obs_non_existent(monkeypatch):
    monkeypatch.setattr(table.Table, "write", _disabled_write)  # just in case

    with pytest.raises(FileExistsError):
        # if ref is empty, no exception is raised
        ref = {
            "obs": [
                {"obsid": 56314, "agasc_id": 2, "status": 0, "comments": "comment"}
            ],
            "bad": [],
            "mags": [],
        }
        update_supplement.update_obs_table(
            pathlib.Path("some_non_existent_file.h5"), ref["obs"]
        )


def test_update_obs_dry_run(monkeypatch):
    # should not write if dry_run==True
    monkeypatch.setattr(table.Table, "write", _disabled_write)
    update_supplement.update_obs_table(
        TEST_DATA_DIR / "agasc_supplement_empty.h5", [], dry_run=True
    )


def test_update_obs_skip(monkeypatch):
    # should not write if there is nothing to write
    monkeypatch.setattr(table.Table, "write", _disabled_write)
    update_supplement.update_obs_table(
        TEST_DATA_DIR / "agasc_supplement_empty.h5", [], dry_run=False
    )


def test_update_obs_blank_slate(monkeypatch, mock_open):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    def mock_write(fname, *args, **kwargs):
        assert kwargs["path"] in ["agasc_versions", "last_updated", "obs"]
        mock_write.calls.append((args, kwargs))
        if kwargs["path"] == "obs":
            a = table.Table(args[0])
            b = table.Table(TEST_DATA[fname]["obs"], dtype=OBS_DTYPE)
            a.sort(keys=["agasc_id", "obsid"])
            b.sort(keys=["agasc_id", "obsid"])
            assert np.all(a == b)

    mock_write.calls = []

    for filename in TEST_YAML:
        monkeypatch.setattr(
            table.Table,
            "write",
            lambda *args, **kwargs: mock_write(filename, *args, **kwargs),  # noqa: B023
        )
        status = update_supplement.parse_args(filename=filename)
        print(filename)
        print(status["obs"])
        update_supplement.update_obs_table(
            TEST_DATA_DIR / "agasc_supplement_empty.h5", status["obs"], dry_run=False
        )
    assert len(mock_write.calls) > 0, "Table.write was never called"


def test_update_obs(monkeypatch, mock_open):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    def mock_write(*args, **kwargs):
        mock_write.calls.append((args, kwargs))
        if "path" in kwargs and kwargs["path"] == "bad":
            mock_write.n_calls += 1
            ref = table.Table()
            raise AssertionError()
        if "path" in kwargs and kwargs["path"] == "obs":
            mock_write.n_calls += 1
            ref = table.Table(
                np.array(
                    [
                        (56311, 563087864, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563088952, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563089432, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563091784, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563092520, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563612488, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563612792, 1, "", "2010:110:18:59:46.600"),
                        (56311, 563617352, 1, "", "2010:110:18:59:46.600"),
                        (56308, 806750112, 0, "", "2010:110:23:23:49.708"),
                        (
                            11849,
                            1019348536,
                            1,
                            "just removed them",
                            "2010:111:12:26:54.536",
                        ),
                        (
                            11849,
                            1019350904,
                            1,
                            "just removed them",
                            "2010:111:12:26:54.536",
                        ),
                        (
                            56314,
                            114950168,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114950584,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114952056,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114952792,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114952824,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114955056,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            114956608,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                        (
                            56314,
                            115347520,
                            1,
                            "removed because I felt like it",
                            "2010:110:14:57:43.442",
                        ),
                    ],
                    dtype=[
                        ("obsid", "<i4"),
                        ("agasc_id", "<i4"),
                        ("status", "<i4"),
                        ("comments", "<U80"),
                        ("mp_starcat_time", "<U21"),
                    ],
                )
            )
            ref.pprint(max_lines=30)
            args[0][ref.dtype.names].pprint(max_lines=30)
            assert np.all(args[0][ref.dtype.names] == ref)

    mock_write.n_calls = 0
    mock_write.calls = []

    filename = "file_4.yml"
    monkeypatch.setattr(table.Table, "write", mock_write)
    status = update_supplement.parse_args(filename=filename)

    print(status["obs"])
    update_supplement.update_obs_table(
        TEST_DATA_DIR / "agasc_supplement.h5", status["obs"], dry_run=False
    )
    assert mock_write.n_calls == 1, "Table.write was never called"


def test_add_bad_star(monkeypatch):
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    bad = [(77073552, 3), (23434, 10), (53475, None)]

    def mock_write(*args, **kwargs):
        mock_write.calls.append((args, kwargs))
        if "path" in kwargs and kwargs["path"] == "bad":
            mock_write.n_calls += 1
            ref = table.Table(
                [
                    {
                        "agasc_id": 77073552,
                        "source": 2,
                    },  # this is there already, not overwriting
                    {"agasc_id": 23434, "source": 10},  # this one is new
                    {
                        "agasc_id": 53475,
                        "source": 3,
                    },  # source is None, so the value is max(source)
                ],
                dtype=BAD_DTYPE,
            )
            print("ref")
            print(ref)
            print("arg")
            print(args[0])
            assert args[0].dtype == ref.dtype
            assert np.all(args[0] == ref)

    mock_write.n_calls = 0
    mock_write.calls = []

    update_supplement.add_bad_star(
        TEST_DATA_DIR / "agasc_supplement.h5", bad, dry_run=True
    )
    assert mock_write.n_calls == 0, "Table.write was called with dry_run=True"

    monkeypatch.setattr(table.Table, "write", mock_write)
    update_supplement.add_bad_star(
        TEST_DATA_DIR / "agasc_supplement.h5", bad, dry_run=False
    )
    assert mock_write.n_calls == 1, "Table.write was never called"


def test_update_mags(monkeypatch):
    from agasc.supplement.utils import MAGS_DTYPE

    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    # this test starts from the agasc supplement in agasc/tests/data/agasc_supplement.h5
    # some values are already there and we update some
    mags = [
        # a duplicate entry that gets ignored (the last one read takes precedence)
        {"agasc_id": 3, "mag_aca": 7.0, "mag_aca_err": 0.5, "last_obs_time": 0.0},
        # new entry
        {"agasc_id": 3, "mag_aca": 8.0, "mag_aca_err": 1.0, "last_obs_time": 0.0},
        # an existing entry being updated
        {
            "agasc_id": 5,
            "mag_aca": 6.0,
            "mag_aca_err": 0.3,
            "last_obs_time": 695001669.184,
        },
        # an entry with no last_obs_time (it is taken from the stars_obs catalog)
        {
            "agasc_id": 114952792,
            "mag_aca": 7.5,
            "mag_aca_err": 0.2,
            "last_obs_time": 388162729.626,
        },
    ]

    def mock_write(*args, **kwargs):
        mock_write.calls.append((args, kwargs))
        if "path" in kwargs and kwargs["path"] == "mags":
            ref = table.Table(
                [
                    # the new entry
                    {
                        "agasc_id": 3,
                        "last_obs_time": 0.0,
                        "mag_aca": 8.0,
                        "mag_aca_err": 1.0,
                    },
                    # the updated entry
                    {
                        "agasc_id": 5,
                        "last_obs_time": 695001669.184,
                        "mag_aca": 6.0,
                        "mag_aca_err": 0.3,
                    },
                    # an entry that was already there
                    {
                        "agasc_id": 114950168,
                        "last_obs_time": 388162729.626,
                        "mag_aca": 7.0,
                        "mag_aca_err": 0.2,
                    },
                    # the new entry with the right last_obs_time
                    {
                        "agasc_id": 114952792,
                        "last_obs_time": 388162729.626,
                        "mag_aca": 7.5,
                        "mag_aca_err": 0.2,
                    },
                ],
                dtype=MAGS_DTYPE,
            )

            print("arg:")
            print(args[0])
            print("ref:")
            print(ref)

            assert len(args[0]) == len(ref)
            assert ref.colnames == args[0].colnames
            assert ref.dtype == args[0].dtype
            assert np.all(np.sort(args[0]["agasc_id"]) == np.sort(ref["agasc_id"]))
            i, j = np.argwhere(args[0]["agasc_id"][None] == ref["agasc_id"][:, None]).T
            assert np.all(args[0][j] == ref[i])
            mock_write.n_calls += 1

    mock_write.n_calls = 0
    mock_write.calls = []

    update_supplement.update_mags_table(
        TEST_DATA_DIR / "agasc_supplement.h5", mags, dry_run=True
    )
    assert mock_write.n_calls == 0, "Table.write was called with dry_run=True"

    monkeypatch.setattr(table.Table, "write", mock_write)
    update_supplement.update_mags_table(
        TEST_DATA_DIR / "agasc_supplement.h5", mags, dry_run=False
    )
    assert mock_write.n_calls == 1, "Table.write was never called"


def test_save_version(monkeypatch):
    # this test takes the following dictionary and passes it to save_version
    # it then checks that a corresponding astropy table with the right structure is created and its
    # write method is called
    import agasc

    versions = {"obs": agasc.__version__, "mags": agasc.__version__}

    def mock_write(*args, **kwargs):
        mock_write.calls.append((args, kwargs))
        assert "format" in kwargs and kwargs["format"] == "hdf5"
        assert "path" in kwargs
        assert kwargs["path"] in ["last_updated", "agasc_versions"]
        assert len(args[0]) == 1
        assert "supplement" in args[0].colnames
        if kwargs["path"] == "agasc_versions":
            for k, _v in versions.items():
                assert k in args[0].colnames
                assert args[0][k][0] == versions[k]

    mock_write.calls = []

    monkeypatch.setattr(table.Table, "write", mock_write)

    from agasc.supplement.utils import save_version

    save_version("test_save_version.h5", ["obs", "mags"])
    assert len(mock_write.calls) > 0, "Table.write was never called"


def test_override(monkeypatch):
    _monkeypatch_star_obs_catalogs_(
        monkeypatch, test_file=TEST_DATA_DIR / "mag-stats.h5", path="/obs_status"
    )
    telem = _monkeypatch_get_telemetry_(
        monkeypatch, test_file=TEST_DATA_DIR / "mag-stats.h5", path="/obs_status/telem"
    )

    # Case 1. There are two previously unknown suspect observations out of 5.
    # - 23682 has only 10 points
    # - 48900 magnitude changes significantly during the observation
    # in the following test cases, they will be marked as suspect to check that the algorithm
    # excludes them.
    agasc_stats, obs_stats, fails = mag_estimate.get_agasc_id_stats(10492752)

    assert len(fails) == 2
    assert len(obs_stats) == 5
    assert agasc_stats["n_obsids_fail"] == 2
    assert agasc_stats["n_obsids_ok"] == 3
    assert (
        agasc_stats["n_obsids_suspect"] == 2
    )  # two suspect count as "fail" in this context
    assert not np.isclose(
        np.mean(telem[np.in1d(telem["obsid"], [12800])]["mags"]), agasc_stats["mag_obs"]
    )
    assert np.isclose(
        np.mean(telem[np.in1d(telem["obsid"], [12800, 23682, 23683])]["mags"]),
        agasc_stats["mag_obs"],
    )

    # Case 2. Four observations (including the suspect) are marked with non-zero status
    # Only the remaining observation should be included
    obs_status_override = {
        ("2018:296:15:53:14.596", 10492752): {"status": 1, "comments": ""},
        ("2021:015:00:01:45.585", 10492752): {"status": 1, "comments": ""},
        ("2021:089:02:48:00.575", 10492752): {"status": 1, "comments": ""},
        ("2021:201:02:58:03.250", 10492752): {"status": 1, "comments": ""},
    }
    agasc_stats, obs_stats, fails = mag_estimate.get_agasc_id_stats(
        10492752, obs_status_override=obs_status_override
    )

    assert len(fails) == 0
    assert agasc_stats["n_obsids_fail"] == 0
    assert agasc_stats["n_obsids_ok"] == 1
    assert (
        agasc_stats["n_obsids_suspect"] == 0
    )  # no fails because all status==1 skipped
    assert np.isclose(
        np.mean(telem[np.in1d(telem["obsid"], [12800])]["mags"]), agasc_stats["mag_obs"]
    )
    assert not np.isclose(
        np.mean(telem[np.in1d(telem["obsid"], [12800, 23682, 23683])]["mags"]),
        agasc_stats["mag_obs"],
    )

    # Case 3:
    # - One of the suspect observations is previously known and marked as OK (status=0)
    # - One other observation is marked as not-OK
    obs_status_override = {
        ("2021:015:00:01:45.585", 10492752): {"status": 0, "comments": ""},
        ("2021:089:02:48:00.575", 10492752): {"status": 1, "comments": ""},
    }
    agasc_stats, obs_stats, fails = mag_estimate.get_agasc_id_stats(
        10492752, obs_status_override=obs_status_override
    )

    assert len(fails) == 1
    assert agasc_stats["n_obsids_fail"] == 1
    assert agasc_stats["n_obsids_ok"] == 3
    assert agasc_stats["n_obsids_suspect"] == 1  # one failed
    assert np.isclose(
        np.mean(telem[np.in1d(telem["obsid"], [12800, 23681, 23683])]["mags"]),
        agasc_stats["mag_obs"],
    )


def _remove_list_duplicates(a_list):
    # removes duplicates while preserving order
    a_list = a_list.copy()
    remove = []
    for i, item in enumerate(a_list):
        if item in a_list[:i]:
            remove.append(i)
    for i in remove[::-1]:
        del a_list[i]
    return a_list


def _monkeypatch_star_obs_catalogs_(monkeypatch, test_file, path):
    tables = [
        "STARS_OBS",
    ]

    res = {t: table.Table.read(test_file, path=f"{path}/cat/{t}") for t in tables}
    for val in res.values():
        val.convert_bytestring_to_unicode()
    res["STARS_OBS"].add_index("agasc_id")
    res["STARS_OBS"].add_index("mp_starcat_time")

    for key, val in res.items():
        monkeypatch.setattr(star_obs_catalogs, key, val)


def _monkeypatch_get_telemetry_(monkeypatch, test_file, path):
    telem = table.Table.read(test_file, path=path)

    def get_telemetry(obs):
        obsid = obs["obsid"]
        if obs["obsid"] in telem["obsid"]:
            return telem[telem["obsid"] == obs["obsid"]]
        raise Exception(f"{obsid=} not in test telemetry")

    monkeypatch.setattr(mag_estimate, "get_telemetry", get_telemetry)

    return telem


def recreate_mag_stats_test_data(filename=TEST_DATA_DIR / "mag-stats.h5"):
    """
    Create data to test mag-stats.

    The produced data includes five observations (all stars for each observation), and a fraction of
    the telemetry for one single star during each of those observations.

    The telemetry is slightly modified so two of those observations are marked as suspect:
        - 23681 has only ten points
        - 48900 magnitude changes significantly during the observation
    """
    from astropy.table import vstack

    if os.path.exists(filename):
        os.unlink(filename)

    star_obs_catalogs.load()
    mp_starcat_time = [
        "2011:288:06:14:49.501",
        "2021:015:00:01:45.585",
        "2021:089:02:48:00.575",
        "2021:201:02:58:03.250",
        "2018:296:15:53:14.596",
    ]
    STARS_OBS = star_obs_catalogs.STARS_OBS[
        np.in1d(star_obs_catalogs.STARS_OBS["mp_starcat_time"], mp_starcat_time)
    ]
    STARS_OBS = STARS_OBS.group_by("agasc_id")
    STARS_OBS.add_index("agasc_id")
    STARS_OBS.write(
        filename,
        path="/obs_status/cat/STARS_OBS",
        serialize_meta=True,
        append=True,
        overwrite=True,
    )

    telem = mag_estimate.get_telemetry_by_agasc_id(10492752)

    # the starting times might be a bit arbitrary. They are chose to remove the first points, which
    # might be in maneuver mode or in acquisition. These times come from the kadi events v1 version,
    # but they do not matter much.
    telem_by_obsid = [
        telem[(telem["obsid"] == 12800) & (telem["times"] > 435047672.0)][:100],
        # only 10 points, excluding the beginning
        telem[(telem["obsid"] == 23681) & (telem["times"] > 727057549.0)][:10],
        telem[(telem["obsid"] == 23682) & (telem["times"] > 733462165.0)][:100],
        telem[(telem["obsid"] == 23683) & (telem["times"] > 743139160.0)][:100],
        telem[(telem["obsid"] == 48900) & (telem["times"] > 656698074.0)][:100],
    ]
    telem_by_obsid[-1]["mags_img"] += 0.01 * np.exp(np.arange(100) / 20)
    telem_by_obsid[-1]["mags"] += 0.01 * np.exp(np.arange(100) / 20)
    t = vstack(telem_by_obsid)

    t.write(filename, path="/obs_status/telem", serialize_meta=True, append=True)


def recreate_test_supplement(supplement_filename=TEST_DATA_DIR / "agasc_supplement.h5"):
    # this is not a test function, but a function to generate the test supplement from scratch
    # whenever it needs updating, so all the data is actually contained in this file
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setitem(__builtins__, "open", _open)
    monkeypatch.setattr(star_obs_catalogs, "STARS_OBS", STARS_OBS)

    with tables.open_file(str(supplement_filename), "w"):
        pass

    print(f"Updating {supplement_filename}")
    status = update_supplement.parse_args(filename="file_0.yml")
    print("obs")
    print(status["obs"])
    update_supplement.update_obs_table(
        supplement_filename, status["obs"], dry_run=False
    )
    print("mags")
    print(status["mags"])
    update_supplement.update_mags_table(
        supplement_filename, status["mags"], dry_run=False
    )
    print("bad")
    print(status["bad"])
    update_supplement.add_bad_star(supplement_filename, status["bad"], dry_run=False)
