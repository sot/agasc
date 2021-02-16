# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import logging
import warnings

from ska_helpers.utils import lru_cache_timed
import tables
from cxotime import CxoTime
from astropy.table import Table

from ..paths import SUPPLEMENT_FILENAME, default_agasc_dir


__all__ = ['get_supplement_table', 'save_version']


logger = logging.getLogger('agasc.supplement')


AGASC_SUPPLEMENT_TABLES = ('mags', 'bad', 'obs', 'last_updated', 'agasc_versions')


@lru_cache_timed(timeout=3600)
def get_supplement_table(name, agasc_dir=None, as_dict=False):
    """Get one of the tables in the AGASC supplement.

    This function gets one of the supplement tables, specified with ``name``:

    - ``bad``: Bad stars (agasc_id, source)
    - ``mags``: Estimated mags (agasc_id, mag_aca mag_aca_err)
    - ``obs``: Star-obsid status for mag estimation (agasc_id, obsid, ok,
      comments)

    This function is cached with a timeout of an hour, so you can call it
    repeatedly with no penalty in performance.

    If ``as_dict=False`` (default) then the table is returned as an astropy
    ``Table``.

    If ``as_dict=True`` then the table is returned as a dict of {key: value}
    pairs. For ``mags`` and ``bad``, the key is ``agasc_id``. For ``obs`` the
    key is the ``(agasc_id, obsid)`` tuple. In all cases the value is a dict
    of the remaining columns.

    :param name: Table name within the AGASC supplement HDF5 file
    :param data_root: directory containing the AGASC supplement HDF5 file
        (default=same directory as the AGASC file)
    :param as_dict: return result as a dictionary (default=False)

    :returns: supplement table as ``Table`` or ``dict``
    """
    agasc_dir = default_agasc_dir() if agasc_dir is None else Path(agasc_dir)

    if name not in AGASC_SUPPLEMENT_TABLES:
        raise ValueError(f"table name must be one of {AGASC_SUPPLEMENT_TABLES}")

    supplement_file = agasc_dir / SUPPLEMENT_FILENAME
    with tables.open_file(supplement_file) as h5:
        try:
            dat = getattr(h5.root, name)[:]
        except tables.NoSuchNodeError:
            warnings.warn(f"No dataset '{name}' in {supplement_file},"
                          " returning empty table")
            dat = []

    if as_dict:
        out = {}
        keys_names = {
            'mags': ['agasc_id'],
            'bad': ['agasc_id'],
            'obs': ['agasc_id', 'obsid']}
        key_names = keys_names[name]
        for row in dat:
            # Make the key, coercing the values from numpy to native Python
            key = tuple(row[nm].item() for nm in key_names)
            if len(key) == 1:
                key = key[0]
            # Make the value from the remaining non-key column names
            out[key] = {nm: row[nm].item() for nm in row.dtype.names if nm not in key_names}
    else:
        out = Table(dat)

    return out


def _load_or_create(filename, table_name):
    try:
        table = Table.read(filename, format='hdf5', path=table_name)
    except OSError as e:
        logger.debug(f'Creating agasc supplement table "{table_name}"')
        if not filename.exists() or str(e) == f'Path {table_name} does not exist':
            table = Table()
        else:
            raise
    return table


def save_version(filename, table_name):
    """Save the version of a supplement table to the "versions" table.

    Along with the version, the time of update is also added to another table called "last_updated"

    Example usage::

        from agasc.supplement.utils import save_version
        save_version('agasc_supplement.h5', mags='4.10.2')

    The different versions can be retrieved from the default supplement::

        from agasc.supplement.utils import get_supplement_table
        versions = get_supplement_table('agasc_versions')

    :param filename: pathlib.Path
    :param table_name: str or list
    """
    if isinstance(table_name, str):
        table_name = [table_name]

    import agasc
    filename = Path(filename)

    versions = _load_or_create(filename, 'agasc_versions')
    last_updated = _load_or_create(filename, 'last_updated')

    time = CxoTime.now()
    time.precision = 0
    table_name.append('supplement')
    for key in table_name:
        logger.debug(f'Adding "{key}" to supplement "agasc_versions" and "last_updated" table')
        versions[key] = [agasc.__version__]
        last_updated[key] = [time.iso]

    versions.write(str(filename), format='hdf5', path='agasc_versions',
                   append=True, overwrite=True)
    last_updated.write(str(filename), format='hdf5', path='last_updated',
                       append=True, overwrite=True)
