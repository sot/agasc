import sys
from itertools import count
import time
import numpy as np
import agasc


def random_ra_dec(nsample):
    x = np.random.uniform(-0.98, 0.98, size=nsample)
    ras = 360 * np.random.random(nsample)
    decs = np.degrees(np.arcsin(x))
    return ras, decs

radius = 2.0
nsample = 200
ras, decs = random_ra_dec(nsample)

print 'get_agasc_cone'
t0 = time.time()
for ra, dec, cnt in zip(ras, decs, count()):
    x = agasc.get_agasc_cone(ra, dec, radius=radius, agasc_file='miniagasc.h5')
    print cnt, len(ras), '\r',
    sys.stdout.flush()
print
print time.time() - t0
