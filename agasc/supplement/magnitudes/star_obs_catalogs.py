import os

import numpy as np
import tables
from astropy import table
from astropy.table import Table, join
from chandra_aca.transform import yagzag_to_pixels
from kadi import commands, events

STARS_OBS = None
"""The table of star observations"""


def get_star_observations(start=None, stop=None, obsid=None):
    """
    Get a table of star observations.

    This is basically the join of kadi.commands.get_observations and kadi.commands.get_starcats,
    with some extra information (pixel row/col, magnitude error).
    """
    join_keys = ["starcat_date", "obsid"]

    observations = Table(commands.get_observations(start=start, stop=stop, obsid=obsid))
    observations = observations[~observations["starcat_date"].mask]
    # the following line removes manual commands
    observations = observations[observations["source"] != "CMD_EVT"]
    catalogs = commands.get_starcats_as_table(
        start=start, stop=stop, obsid=obsid, unique=True
    )
    catalogs = catalogs[np.in1d(catalogs["type"], ["BOT", "GUI"])]
    star_obs = join(observations, catalogs, keys=join_keys)
    star_obs.rename_columns(["id", "starcat_date"], ["agasc_id", "mp_starcat_time"])
    star_obs["row"], star_obs["col"] = yagzag_to_pixels(
        star_obs["yang"], star_obs["zang"]
    )

    # Add mag_aca_err column
    filename = os.path.join(os.environ["SKA"], "data", "agasc", "proseco_agasc_1p7.h5")
    with tables.open_file(filename) as h5:
        agasc_ids = h5.root.data.col("AGASC_ID")
        mag_errs = h5.root.data.col("MAG_ACA_ERR") * 0.01

    tt = Table([agasc_ids, mag_errs], names=["agasc_id", "mag_aca_err"])
    star_obs = table.join(star_obs, tt, keys="agasc_id")

    star_obs.add_index(["mp_starcat_time"])

    max_time = events.dwells.all().latest("tstart").stop
    star_obs = star_obs[star_obs["obs_start"] <= max_time]

    return star_obs


def load(tstop=None):
    """
    Populate this module's global variable STARS_OBS.
    """
    global STARS_OBS
    if STARS_OBS is None:
        STARS_OBS = get_star_observations(stop=tstop)
