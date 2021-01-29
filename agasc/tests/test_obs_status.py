import pytest
import io
import numpy as np
import pathlib
from astropy import table
import tables

from agasc.supplement.magnitudes import star_obs_catalogs
from agasc.scripts import update_obs_status

TEST_DATA_DIR = pathlib.Path(__file__).parent / 'data'

STARS_OBS = np.array([(56314,  114950168), (56314,  114950584), (56314,  114952056),
                      (56314,  114952792), (56314,  114952824), (56314,  114955056),
                      (56314,  114956608), (56314,  115347520), (56312,  357045496),
                      (56312,  357049064), (56312,  357051640), (56312,  357054680),
                      (56312,  358220224), (56312,  358222256), (56312,  358224400),
                      (56312,  358757768), (56313,  441853632), (56313,  441854760),
                      (56313,  441855776), (56313,  441856032), (56313,  441856400),
                      (56313,  441980072), (56313,  491391592), (56313,  491394504),
                      (56311,  563087864), (56311,  563088952), (56311,  563089432),
                      (56311,  563091784), (56311,  563092520), (56311,  563612488),
                      (56311,  563612792), (56311,  563617352), (56310,  624826320),
                      (56310,  624826464), (56310,  624828488), (56310,  624831328),
                      (56310,  624831392), (56310,  624954248), (56310,  624956216),
                      (56310,  625476960), (12203,  697581832), (12203,  697963056),
                      (12203,  697963288), (12203,  697970576), (12203,  697973824),
                      (56308,  732697144), (56308,  732698416), (56309,  762184312),
                      (56309,  762184768), (56309,  762185584), (56309,  762186016),
                      (56309,  762186080), (56309,  762191224), (56309,  762579584),
                      (56309,  762581024), (56308,  806748432), (56308,  806748880),
                      (56308,  806750112), (56308,  806750408), (56308,  806750912),
                      (56308,  806751424), (56306,  956708808), (56306,  957219128),
                      (56306,  957221200), (56306,  957222432), (56306,  957229080),
                      (56306,  957230976), (56306,  957233920), (56306,  957369088),
                      (11849, 1019347720), (11849, 1019348536), (11849, 1019350904),
                      (11849, 1019354232), (11849, 1019357032), (11980, 1198184872),
                      (11980, 1198190648), (11980, 1198190664), (11980, 1198191400),
                      (11980, 1198192456)],
                     dtype={
                         'names': ['obsid', 'agasc_id'],
                         'formats': ['<i8', '<i8'],
                         'offsets': [0, 16],
                         'itemsize': 264
                     })


TEST_YAML = {
    'file_4.yml': """
                  obs:
                    - obsid: 56314
                      ok: false
                      comments: removed because I felt like it
                      agasc_id: [114950168, 114950584, 114952056, 114952792,
                                 114952824, 114955056, 114956608, 115347520]
                  bad:
                    77073552: null
                    23434: 10
                  """,
    'file_5.yml': """
                  obs:
                    - obsid: 56314
                      ok: false
                      comments: removed because I felt like it
                  bad:
                    77073552: null
                    23434: 10
                  """,
    'file_6.yml': """
                  obs:
                    - obsid: 56314
                      ok: false
                      agasc_id: 114950168
                  """,
    'file_7.yml': """
                  obs:
                    - obsid: 56311
                      ok: false
                    - obsid: 56308
                      ok: true
                      agasc_id: [806750112]
                    - obsid: 11849
                      ok: false
                      agasc_id: [1019348536, 1019350904]
                      comments: just removed them
    """,
    'file_1.yml': """
                  obs:
                    - obsid: 80616784
                      ok: true
                  bad:
                    77073552: null
                    23434: 10
                  """,
    'file_2.yml': """
                  bad:
                    77073552: null
                    23434: 10
                  """,
    'file_3.yml': """
              obs:
                - obsid: 80616784
                  ok: true
              """,
}


TEST_DATA = {
    'file_1.yml': {'obs': {}, 'bad': {77073552: None, 23434: 10}},
    'file_2.yml': {'obs': {}, 'bad': {77073552: None, 23434: 10}},
    'file_3.yml': {'obs': {}, 'bad': {}},
    'file_4.yml': {
        'obs': {
            (56314, 114950168): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114950584): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952056): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952792): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952824): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114955056): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114956608): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 115347520): {'ok': False, 'comments': 'removed because I felt like it'}
        },
        'bad': {77073552: None, 23434: 10}},
    'file_5.yml': {
        'obs': {
            (56314, 114950168): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114950584): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952056): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952792): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114952824): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114955056): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 114956608): {'ok': False, 'comments': 'removed because I felt like it'},
            (56314, 115347520): {'ok': False, 'comments': 'removed because I felt like it'}
        },
        'bad': {77073552: None, 23434: 10}},
    'file_6.yml': {
        'obs': {(56314, 114950168): {'ok': False, 'comments': ''}},
        'bad': {}},
    'file_7.yml': {
        'obs': {
            (56311, 563087864): {'ok': False, 'comments': ''},
            (56311, 563088952): {'ok': False, 'comments': ''},
            (56311, 563089432): {'ok': False, 'comments': ''},
            (56311, 563091784): {'ok': False, 'comments': ''},
            (56311, 563092520): {'ok': False, 'comments': ''},
            (56311, 563612488): {'ok': False, 'comments': ''},
            (56311, 563612792): {'ok': False, 'comments': ''},
            (56311, 563617352): {'ok': False, 'comments': ''},
            (56308, 806750112): {'ok': True, 'comments': ''},
            (11849, 1019348536): {'ok': False, 'comments': 'just removed them'},
            (11849, 1019350904): {'ok': False, 'comments': 'just removed them'}
        },
        'bad': {}}
    }


def _open(filename):
    return io.StringIO(TEST_YAML[filename])


def test_parse_obs_status_file(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)

    with pytest.raises(RuntimeError, match=r"catalog"):
        update_obs_status._parse_obs_status_file('file_4.yml')

    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    for filename in TEST_YAML:
        data = update_obs_status._parse_obs_status_file(filename)
        assert data == TEST_DATA[filename], \
            f'_parse_obs_status_file("{filename}") == TEST_DATA["{filename}"]'


def test_parse_obs_status_args_file(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    for filename in TEST_YAML:
        ref_obs_status_override, ref_bad = TEST_DATA[filename]
        obs_status_override, bad = update_obs_status._parse_obs_status_args(
            filename=filename
        )
        assert obs_status_override == ref_obs_status_override, \
            f'_parse_obs_status_args("{filename}")[0] == TEST_DATA["{filename}"][0]'
        assert bad == ref_bad, \
            f'_parse_obs_status_args("{filename}")[1] == TEST_DATA["{filename}"][1]'


def test_parse_obs_status_args_bad(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    #######################
    # specifying bad stars
    #######################

    status = update_obs_status._parse_obs_status_args(
        bad_star=1, bad_star_source=2
    )
    assert status['obs'] == {}
    assert status['bad'] == {1: 2}

    # bad star can be a list
    status = update_obs_status._parse_obs_status_args(
        bad_star=[1, 2], bad_star_source=3
    )
    assert status['obs'] == {}
    assert status['bad'] == {1: 3, 2: 3}

    # if you specify bad_star, you must specify bad_star_source
    with pytest.raises(RuntimeError, match=r"specify bad_star_source"):
        update_obs_status._parse_obs_status_args(
            bad_star=1
        )


def test_parse_obs_status_args_obs(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    #######################
    # specifying obs status
    #######################

    status = update_obs_status._parse_obs_status_args(
        obsid=56314, status=False, comments='some comment'
    )
    ref = {
        'obs': {
            (56314, 114950168): {'ok': False, 'comments': 'some comment'},
            (56314, 114950584): {'ok': False, 'comments': 'some comment'},
            (56314, 114952056): {'ok': False, 'comments': 'some comment'},
            (56314, 114952792): {'ok': False, 'comments': 'some comment'},
            (56314, 114952824): {'ok': False, 'comments': 'some comment'},
            (56314, 114955056): {'ok': False, 'comments': 'some comment'},
            (56314, 114956608): {'ok': False, 'comments': 'some comment'},
            (56314, 115347520): {'ok': False, 'comments': 'some comment'}
        },
        'bad': {}
    }
    assert status == ref

    # comments are optional
    status = update_obs_status._parse_obs_status_args(
        obsid=56314, status=False
    )
    ref = {
        'obs': {
            (56314, 114950168): {'ok': False, 'comments': ''},
            (56314, 114950584): {'ok': False, 'comments': ''},
            (56314, 114952056): {'ok': False, 'comments': ''},
            (56314, 114952792): {'ok': False, 'comments': ''},
            (56314, 114952824): {'ok': False, 'comments': ''},
            (56314, 114955056): {'ok': False, 'comments': ''},
            (56314, 114956608): {'ok': False, 'comments': ''},
            (56314, 115347520): {'ok': False, 'comments': ''}},
        'bad': {}
    }
    assert status == ref

    # OBSID does not exist, so there are no stars in it
    status = update_obs_status._parse_obs_status_args(
        obsid=1, status=False
    )
    assert status == {'obs': {}, 'bad': {}}

    # optional agasc_id can be int or list
    status = update_obs_status._parse_obs_status_args(
        obsid=1, status=True, agasc_id=[2], comments='comment'
    )
    ref = {'obs': {(1, 2): {'ok': True, 'comments': 'comment'}},
           'bad': {}}
    assert status == ref

    status = update_obs_status._parse_obs_status_args(
        obsid=1, status=True, agasc_id=2, comments='comment'
    )
    ref = {'obs': {(1, 2): {'ok': True, 'comments': 'comment'}},
           'bad': {}}
    assert status == ref


def test_parse_obs_status_args(monkeypatch):
    import copy
    monkeypatch.setitem(__builtins__, 'open', _open)

    # calling function before catalog is initialized gives an exception
    with pytest.raises(RuntimeError, match=r"catalog"):
        _ = update_obs_status._parse_obs_status_args('file_5.yml')

    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    filename = 'file_4.yml'

    # can not specify bad star with different source in the file and in args.
    with pytest.raises(RuntimeError, match=r"name collision"):
        _ = update_obs_status._parse_obs_status_args(
            filename=filename,
            bad_star=23434,
            bad_star_source=12
        )

    # can specify bad star in the file and in args if the source is the same.
    status = update_obs_status._parse_obs_status_args(
        filename=filename,
        bad_star=23434,
        bad_star_source=10
    )
    ref = copy.deepcopy(TEST_DATA[filename])
    assert ref == status

    # if there are no name conflicts, args and file are merged
    status = update_obs_status._parse_obs_status_args(
        filename=filename,
        obs=56309,
        agasc_id=[762184312, 762184768, 762185584, 762186016],
        status=False,
        bad_star=[1, 2],
        bad_star_source=1000
    )
    ref = update_obs_status._parse_obs_status_args(
        filename=filename
    )
    ref_2 = update_obs_status._parse_obs_status_args(
        obs=56309,
        agasc_id=[762184312, 762184768, 762185584, 762186016],
        status=False,
        bad_star=[1, 2],
        bad_star_source=1000
    )
    ref['obs'].update(ref_2['obs'])
    ref['bad'].update(ref_2['bad'])

    assert status == ref


def _disabled_write(*args, **kwargs):
    raise Exception('Tried to write file when it should not')


def test_update_obs_non_existent():
    with pytest.raises(FileExistsError):
        update_obs_status.update_obs_table('some_non_existent_file.h5', {})


def test_update_obs_dry_run(monkeypatch):
    # should not write if dry_run==True
    monkeypatch.setattr(table.Table, 'write', _disabled_write)
    update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement_empty.h5',
                                        {},
                                        dry_run=True)


def test_update_obs_skip(monkeypatch):
    # should not write if there is nothing to write
    monkeypatch.setattr(table.Table, 'write', _disabled_write)
    update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement_empty.h5',
                                        {},
                                        dry_run=False)


def test_update_obs_blank_slate(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    def mock_write(fname, *args, **kwargs):
        obs_status = {
            (r['obsid'], r['agasc_id']): {'ok': r['ok'], 'comments': r['comments']}
            for r in args[0]
        }
        assert obs_status == TEST_DATA[fname]['obs']

    for filename in TEST_YAML:
        monkeypatch.setattr(table.Table,
                            'write',
                            lambda *args, **kwargs: mock_write(filename, *args, **kwargs))
        status = update_obs_status._parse_obs_status_args(filename=filename)
        update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement_empty.h5',
                                            status['obs'],
                                            dry_run=False)


def test_update_obs(monkeypatch):
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    def mock_write(*args, **kwargs):
        ref = table.Table(np.array([(56311,  563087864, 0, ''), (56311,  563088952, 0, ''),
                                    (56311,  563089432, 0, ''), (56311,  563091784, 0, ''),
                                    (56311,  563092520, 0, ''), (56311,  563612488, 0, ''),
                                    (56311,  563612792, 0, ''), (56311,  563617352, 0, ''),
                                    (56308,  806750112, 1, ''),
                                    (11849, 1019348536, 0, 'just removed them'),
                                    (11849, 1019350904, 0, 'just removed them'),
                                    (56314,  114950168, 0, 'removed because I felt like it'),
                                    (56314,  114950584, 0, 'removed because I felt like it'),
                                    (56314,  114952056, 0, 'removed because I felt like it'),
                                    (56314,  114952792, 0, 'removed because I felt like it'),
                                    (56314,  114952824, 0, 'removed because I felt like it'),
                                    (56314,  114955056, 0, 'removed because I felt like it'),
                                    (56314,  114956608, 0, 'removed because I felt like it'),
                                    (56314,  115347520, 0, 'removed because I felt like it')],
                                   dtype=[('obsid', '<i8'), ('agasc_id', '<i8'),
                                          ('ok', '<u8'), ('comments', '<U30')])
                          )
        assert np.all(args[0] == ref)

    filename = 'file_4.yml'
    monkeypatch.setattr(table.Table, 'write', mock_write)
    status = update_obs_status._parse_obs_status_args(filename=filename)
    update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement.h5',
                                        status['obs'],
                                        dry_run=False)


def recreate_test_supplement():
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    monkeypatch.setitem(__builtins__, 'open', _open)
    monkeypatch.setattr(star_obs_catalogs, 'STARS_OBS', STARS_OBS)

    with tables.open_file(str(TEST_DATA_DIR / 'agasc_supplement.h5'), 'w'):
        pass

    obs_status_override, bad_stars = update_obs_status._parse_obs_status_args(filename='file_7.yml')
    update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement.h5',
                                        obs_status_override,
                                        dry_run=False)

    obs_status_override, bad_stars = update_obs_status._parse_obs_status_args(filename='file_4.yml')
    update_obs_status.update_obs_table(TEST_DATA_DIR / 'agasc_supplement.h5',
                                        obs_status_override,
                                        dry_run=False)

