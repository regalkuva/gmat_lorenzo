# THRUST in GEO

from matplotlib import pyplot as plt
import numpy as np

from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time, TimeDelta
from astropy import units as u

from poliastro.bodies import Earth, Moon
from poliastro.constants import rho0_earth, H0_earth

from poliastro.core.elements import rv2coe
from poliastro.core.perturbations import (J2_perturbation, atmospheric_drag_exponential, third_body)
from poliastro.core.propagation import func_twobody 
from poliastro.ephem import build_ephem_interpolant
from poliastro.plotting import OrbitPlotter2D, OrbitPlotter3D
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.util import norm, time_range, Time


from poliastro.twobody.thrust import change_ecc_inc

ecc_0 = .3
ecc_f = 0
a = 42164
inc_0 = 0
inc_f = 20 * u.deg
argp = 0
f = 2.4e-6 * (u.km / u.s**2)

k = Earth.k.to(u.km**3 / u.s**2).value
orb0 = Orbit.from_classical(
    Earth,
    a * u.km,
    ecc_0 * u.one,
    inc_0 * u.deg,
    0 * u.deg,
    argp * u.deg,
    0 * u.deg,
    epoch = Time(0, format = "jd", scale = "tdb")

)

a_d, delta_V, t_f = change_ecc_inc(orb0, ecc_f, inc_f, f) 

def f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = a_d(t0, state, k)
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

tofs = TimeDelta(np.linspace(0, t_f, num = 1000))

ephem = orb0.to_ephem(EpochsArray(orb0.epoch + tofs, method = CowellPropagator(rtol = 1e-6, f = f)))

# plot for Jupyter
frame = OrbitPlotter3D()
frame.set_attractor(Earth)
frame.plot_ephem(ephem, label = "eccentricity and inclination maneuver in GEO")
