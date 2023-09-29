import time
import numpy as np
import argparse
from pathlib import Path
import agasc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("agasc_file", type=str)
    parser.add_argument("--n-cone", type=int, default=500)
    parser.add_argument("--n_get_star", type=int, default=200)
    parser.add_argument("--n_get_stars", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Cache")
    return parser


def get_ra_decs():
    np.random.seed(10)
    n_sim = 500
    # Generate random points on a sphere
    ras = 360 * np.random.uniform(0, 1, n_sim)
    decs = np.arccos(2 * np.random.uniform(0, 1, n_sim) - 1) * 180 / np.pi - 90
    return ras, decs


def time_get_agasc_cone(agasc_file, ras, decs, n_cone, cache=False):
    kwargs = dict(radius=1.0, agasc_file=agasc_file)
    if cache:
        kwargs["cache"] = True

    time0 = time.time()
    for ra, dec in zip(ras[:n_cone], decs[:n_cone]):
        stars = agasc.get_agasc_cone(ra, dec, **kwargs)
        assert len(stars) > 5
    print(f"agasc_get_cone: time for {n_cone} queries: {time.time() - time0:.2f} s")


def time_get_star(agasc_file, ra, dec, radius=2.0, n_get_star=200):
    stars = agasc.get_agasc_cone(10, 20, radius=radius, agasc_file=agasc_file)
    stars = stars[:n_get_star]

    time0 = time.time()
    for star in stars:
        star_new = agasc.get_star(star["AGASC_ID"], agasc_file=agasc_file)
        assert star_new["AGASC_ID"] == star["AGASC_ID"]
    print(f"get_star: time for {len(stars)} queries: {time.time() - time0:.2f} s")


def time_get_stars(agasc_file, ras, decs, radius=2.0, n_get_star=200, n_get_stars=1):
    stars_list = []
    for ra, dec in zip(ras[:n_get_stars], decs[:n_get_stars]):
        stars = agasc.get_agasc_cone(ra, dec, radius=radius, agasc_file=agasc_file)
        stars_list.append(stars[:n_get_star])

    time0 = time.time()
    for stars in stars_list:
        star_new = agasc.get_stars(stars["AGASC_ID"], agasc_file=agasc_file)
        assert np.all(star_new["AGASC_ID"] == stars["AGASC_ID"])
    print(
        f"get_stars: time for {len(stars_list)} queries of {len(stars)} stars: "
        f" {time.time() - time0:.2f} s"
    )


def main():
    print(f"AGASC module version: {agasc.__version__}")
    print(f"AGASC module file: {agasc.__file__}")
    parser = get_parser()
    args = parser.parse_args()
    agasc_file = Path(args.agasc_file).expanduser()
    print(f"AGASC file: {agasc_file}")
    ras, decs = get_ra_decs()
    time_get_agasc_cone(agasc_file, ras, decs, args.n_cone, args.cache)
    time_get_star(agasc_file, 10, 20, 2.0, args.n_get_star)
    time_get_stars(agasc_file, ras, decs, 2.0, args.n_get_star, args.n_get_stars)


if __name__ == "__main__":
    main()
