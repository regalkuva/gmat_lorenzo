## POLIASTRO MAIN FEATURES

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import time_range, Time


# intial parameters of RHW
a    = 6800 * u.kilometer
ecc  = .0016628 * u.one
inc  = 97.4864 * u.deg
raan = 39.164 * u.deg
argp = 325.3203 * u.deg
nu   = 126.202 * u.deg
epoch = Time("2018-11-30 03:53:03.550", scale="utc")


# Definition of the initial RHW orbit
rhw0 = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)


# 2D plot of RHW orbit
from poliastro.plotting import OrbitPlotter2D

op = OrbitPlotter2D()
op.plot(rhw0)


# Keplerian propagation
rhw_60min = rhw0.propagate(60*u.min)


# Sampling of orbits during a time range (it can be epochs array, epoch bounds or true anomaly bounds)
from poliastro.twobody.sampling import EpochsArray, EpochBounds, TrueAnomalyBounds

start_date = Time("2018-11-30 03:53:03.550", scale="utc")
end_date   = Time("2023-10-22 20:38:42.288", scale="utc")

ephem = rhw0.to_ephem(strategy = EpochBounds(min_epoch = start_date, max_epoch = end_date))


# Non-Keplerian propagation
from numba import njit
import numpy as np
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody 
from poliastro.twobody.propagation import CowellPropagator

def j2pert(t0, u_, k):
    du_kep = func_twobody(t0, u_, k)
    ax, ay, az = J2_perturbation(
        t0, u_, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad

rhw1 = rhw0.propagate(10 * u.day, method = CowellPropagator(f=j2pert))







