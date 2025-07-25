# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tables
from astropy.table import Table, unique, vstack
from cxotime import CxoTime
from ska_helpers.utils import lru_cache_timed

from agasc.paths import SUPPLEMENT_FILENAME, default_agasc_dir

__all__ = [
    "get_supplement_table",
    "save_version",
    "update_mags_table",
    "update_obs_table",
    "add_bad_star",
]


logger = logging.getLogger("agasc.supplement")


AGASC_SUPPLEMENT_TABLES = ("mags", "bad", "obs", "last_updated", "agasc_versions")


BAD_DTYPE = np.dtype([("agasc_id", np.int32), ("source", np.int16)])

MAGS_DTYPE = np.dtype(
    [
        ("agasc_id", np.int32),
        ("mag_aca", np.float32),
        ("mag_aca_err", np.float32),
        ("last_obs_time", np.float64),
    ]
)

OBS_DTYPE = np.dtype(
    [
        ("mp_starcat_time", "<U21"),
        ("agasc_id", np.int32),
        ("obsid", np.int32),
        ("status", np.int32),
        ("comments", "<U80"),
    ]
)


COLUMN_DESCRIPTION = {
    "agasc_id": "The unique AGASC ID.",
    "source": "Bad star disposition source.",
    "mag_aca": "Star magnitude determined with ACA.",
    "mag_aca_err": "Star magnitude uncertainty determined with ACA.",
    "last_obs_time": "mp_starcat_time of the last observation of a star",
    "mp_starcat_time": (
        "The time at which the star catalog is commanded (from kadi.commands)"
    ),
    "obsid": (
        "The OBSID at the time the star catalog is commanded (from kadi.commands)."
    ),
    "status": (
        "Flag to tell include/excude the observation when estimating magnitude (0 means"
        ' "include")'
    ),
    "comments": "",
}


@lru_cache_timed(timeout=3600)
def get_supplement_table(
    name: str,
    agasc_dir: Optional[str] = None,
    as_dict: bool = False,
) -> Table | dict:
    """Get one of the tables in the AGASC supplement.

    This function gets one of the supplement tables, specified with ``name``:

    - ``bad``: Bad stars (agasc_id, source)
    - ``mags``: Estimated mags (agasc_id, mag_aca mag_aca_err)
    - ``obs``: Star-observation status for mag estimation (mp_starcat_time, agasc_id, obsid, status,
      comments)

    This function is cached with a timeout of an hour, so you can call it
    repeatedly with no penalty in performance.

    If ``as_dict=False`` (default) then the table is returned as an astropy
    ``Table``.

    If ``as_dict=True`` then the table is returned as a dict of {key: value}
    pairs. For ``mags`` and ``bad``, the key is ``agasc_id``. For ``obs`` the
    key is the ``(agasc_id, mp_starcat_time)`` tuple. In all cases the value is a dict
    of the remaining columns.

    Parameters
    ----------
    name : str
        Table name within the AGASC supplement HDF5 file
    agasc_dir : str, optional
        directory containing the AGASC supplement HDF5 file
        (default=same directory as the AGASC file)
    as_dict : bool
        return result as a dictionary (default=False)

    Returns
    -------
    supplement table as ``Table`` or ``dict``
    """
    agasc_dir = default_agasc_dir() if agasc_dir is None else Path(agasc_dir)

    dtypes = {"bad": BAD_DTYPE, "mags": MAGS_DTYPE, "obs": OBS_DTYPE}

    if name not in AGASC_SUPPLEMENT_TABLES:
        raise ValueError(f"table name must be one of {AGASC_SUPPLEMENT_TABLES}")

    supplement_file = agasc_dir / SUPPLEMENT_FILENAME
    with tables.open_file(supplement_file) as h5:
        try:
            dat = getattr(h5.root, name)[:]
        except tables.NoSuchNodeError:
            warnings.warn(
                f"No dataset '{name}' in {supplement_file}, returning empty table"
            )
            dat = np.array([], dtype=dtypes.get(name, []))

    if as_dict:
        if name in ["agasc_versions", "last_updated"]:
            out = {name: dat[0][name] for name in dat.dtype.names} if len(dat) else {}
        else:
            out = {}
            keys_names = {
                "mags": ["agasc_id"],
                "bad": ["agasc_id"],
                "obs": ["agasc_id", "mp_starcat_time"],
            }
            key_names = keys_names[name]
            for row in dat:
                # Make the key, coercing the values from numpy to native Python
                key = tuple(row[nm].item() for nm in key_names)
                if len(key) == 1:
                    key = key[0]
                # Make the value from the remaining non-key column names
                out[key] = {
                    nm: row[nm].item() for nm in row.dtype.names if nm not in key_names
                }
    else:
        out = Table(dat)
        index = {agasc_id: idx for idx, agasc_id in enumerate(out["agasc_id"])}
        out.meta["index"] = index

    return out


def save_version(filename, table_name):
    """Save the version of a supplement table to the "versions" table.

    Along with the version, the time of update is also added to another table called "last_updated"

    Example usage::

        from agasc.supplement.utils import save_version
        save_version('agasc_supplement.h5', mags='4.10.2')

    The different versions can be retrieved from the default supplement::

        from agasc.supplement.utils import get_supplement_table
        versions = get_supplement_table('agasc_versions')

    Parameters
    ----------
    filename
        pathlib.Path
    table_name
        str or list
    """
    import agasc  # noqa: PLC0415

    if isinstance(table_name, str):
        table_name = [table_name]

    filename = Path(filename)

    versions = _get_table(filename, "agasc_versions", create=True)
    last_updated = _get_table(filename, "last_updated", create=True)

    time = CxoTime.now()
    time.precision = 0
    table_name.append("supplement")
    for key in table_name:
        logger.debug(
            f'Adding "{key}" to supplement "agasc_versions" and "last_updated" table'
        )
        versions[key] = [agasc.__version__]
        last_updated[key] = [time.iso]

    versions.write(
        str(filename), format="hdf5", path="agasc_versions", append=True, overwrite=True
    )
    last_updated.write(
        str(filename), format="hdf5", path="last_updated", append=True, overwrite=True
    )


def update_table(filename, table, path, dtype, keys, dry_run=False, create=False):
    """
    Update a table of the AGASC supplement.

    This overwrites all rows already in the supplement table and appends that rows not yet in the
    supplement table.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file
    table : table.Table | None
        Table update values
    path : str
        Path of the table in the supplement ('obs', 'mags')
    dtype : np.dtype
        Dtype of the table. Used only if the table is not in the supplement already.
        If it is not given, and the table does not exist in the supplement,
        an exception will be raised.
    keys : list
        List of columns that uniquely identify a row in the table.
        In case of duplicates, the last appearance is kept.
    dry_run: bool
        Do not save the table.
    create : bool
        Create a supplement file if it does not exist
    """
    filename = Path(filename)

    if len(table) == 0:
        logger.info("Nothing to update")
        return

    suppl_table = _get_table(filename, path, dtype, create=create)
    table = Table(table if len(table) else None, dtype=dtype)
    table = unique(table, keys=keys, keep="last")

    # for now I require no masked elements.
    if table.mask is not None and np.any(table.mask):
        masked = [name for name in table.colnames if np.any(table[name].mask)]
        raise Exception(
            (
                '"{path}" table should have no masked elements, '
                "but element in these columns are masked: "
            ),
            ", ".join(masked),
        )

    # entries already in supplement
    intersect = table[keys[0]][None, :] == suppl_table[keys[0]][:, None]
    for key in keys[1:]:
        intersect &= table[key][None, :] == suppl_table[key][:, None]
    i, j = np.argwhere(intersect).T

    # entries not yet in supplement
    append = ~np.in1d(table[keys[0]], suppl_table[keys[0]])
    for key in keys[1:]:
        append |= ~np.in1d(table[key], suppl_table[key])
    k = np.argwhere(append).T

    if not len(i) and not len(k[0]):
        return

    logger.info(f'updating "{path}" table in {filename}')

    if len(i):
        suppl_table[i] = table[j]
    if len(k[0]):
        suppl_table = vstack([suppl_table, table[k[0]]])

    if not dry_run:
        suppl_table.write(
            str(filename), format="hdf5", path=path, append=True, overwrite=True
        )
        save_version(filename, path)
    else:
        logger.info("dry run, not saving anything")


def _get_table(filename, path, dtype=None, create=False):
    """
    Gets a table from an HDF5 file.

    If the file does not exist, it either raises an exception or issues a warning, depending on the
    input arguments. If the table does not exist, it returns a newly created table.

    Parameters
    ----------
    filename : str or pathlib.Path
        The name of the HDF5 file.
    path : str
        Path of the table in the supplement ('obs', 'mags')
    dtype : np.dtype
        Numpy dtype of the table. If the table exists, the dtype is ignored. If the table does not
        exist and no dtype is given, an exception will be raised.
    create : bool
        Issues a warning if the supplement file does not exist

    Returns
    -------
    table.Table
    """
    filename = Path(filename)
    if not filename.exists():
        if not create:
            raise FileExistsError(filename)
        logger.warning(
            f"AGASC supplement file does not exist: {filename}. Will create it."
        )

    try:
        result = Table.read(filename, format="hdf5", path=path)
    except OSError as e:
        if not filename.exists() or str(e) == f"Path {path} does not exist":
            result = Table(dtype=dtype)
        else:
            raise

    return result


def add_bad_star(filename, bad, dry_run=False, create=False):
    """
    Update the 'bad' table of the AGASC supplement.

    Parameters
    ----------
    filename : pathlib.Path
        AGASC supplement filename.
    bad : list
        List of pairs (agasc_id, source)
    dry_run : bool
        Do not save the table.
    create : bool
        Create a supplement file if it does not exist
    """
    filename = Path(filename)

    if not len(bad):
        logger.info("Nothing to update")
        return

    logger.info(f'updating "bad" table in {filename}')

    dat = _get_table(filename, "bad", BAD_DTYPE, create=create)
    default = dat["source"].max() + 1 if len(dat) > 0 else 1

    bad = [
        [agasc_id, default if source is None else source] for agasc_id, source in bad
    ]

    bad_star_ids, bad_star_source = np.array(bad).astype(int).T

    update = False
    for agasc_id, source in zip(bad_star_ids, bad_star_source):
        if agasc_id not in dat["agasc_id"]:
            dat.add_row((agasc_id, source))
            logger.info(f"Appending {agasc_id=} with {source=}")
            update = True
    if not update:
        return

    logger.info("")
    logger.info("IMPORTANT:")
    logger.info("Edit following if source ID is new:")
    logger.info(
        "  https://github.com/sot/agasc/wiki/Add-bad-star-to-AGASC-supplement-manually"
    )
    logger.info("")
    logger.info("The wiki page also includes instructions for test, review, approval")
    logger.info("and installation.")
    if not dry_run:
        dat.write(str(filename), format="hdf5", path="bad", append=True, overwrite=True)
        save_version(filename, "bad")


def update_obs_table(filename, obs, dry_run=False, create=False):
    """
    Update the 'obs' table of the AGASC supplement.

    Parameters
    ----------
    filename :
        AGASC supplement filename
    obs : list of dict or table.Table
        Dictionary with status flag for specific observations.
        list entries are dictionaries like::
        {'agasc_id': 1,
        'mp_starcat_time': '2009:310:17:26:44.706',
        'obsid': 0,
        'status': 0,
        'comments': 'some comment'}
        All the keys are optional except 'status', as long as the observations are uniquely defined.
        If 'agasc_id' is not given, then it applies to all stars in that observation.
    dry_run : bool
        Do not save the table.
    create : bool
        Create a supplement file if it does not exist
    """

    update_table(
        filename,
        obs,
        "obs",
        OBS_DTYPE,
        keys=["agasc_id", "mp_starcat_time"],
        dry_run=dry_run,
        create=create,
    )


def update_mags_table(filename, mags, dry_run=False, create=False):
    """
    Update the 'mags' table of the AGASC supplement.

    Parameters
    ----------
    filename :
        AGASC supplement filename
    mags : list of dict or table.Table
        list entries are dictionaries like::
        {'agasc_id': 1,
        'mag_aca': 9.,
        'mag_aca_err': 0.2,
        'last_obs_time': 541074324.949}
    dry_run : bool
        Do not save the table.
    """
    update_table(
        filename,
        mags,
        "mags",
        MAGS_DTYPE,
        keys=["agasc_id"],
        dry_run=dry_run,
        create=create,
    )
