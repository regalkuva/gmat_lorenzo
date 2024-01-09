from astropy import units as u
from astropy import time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import Time

import multiprocessing as mp


start_date = Time('2023-01-01 00:00:00.000', scale='utc')

def create_constellation(epoch):
    a = Earth.R + 380*u.km
    ecc = 0.001 * u.one
    inc = 96.9172647751491 * u.deg
    argp = 0 * u.deg

    return [Orbit.from_classical(Earth, a, ecc, inc, 120*plane*u.deg, argp, 4*sat*u.deg, start_date)\
            for plane in range(0,32) for sat in range(0,50)]


def propagate(sat):
    return sat.propagate(10 * u.day)       


epoch = time.Time("2018-04-25 11:00")    
spx = create_constellation(epoch)
    
p = mp.Pool(4)
result = p.map(propagate,spx)
p.close()
p.join()