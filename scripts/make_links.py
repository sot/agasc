import os
from pathlib import Path
from Ska.Shell import bash


for agasc_dir in Path('agasc').glob("*"):
    updir = agasc_dir.name.upper()
    if not os.path.exists(updir):
        os.makedirs(updir)
    for ffile in agasc_dir.glob("*.fit"):
        os.chdir(updir)
        newname = ffile.name.upper()
        os.symlink((".." / ffile), newname)
        os.chdir("..")
