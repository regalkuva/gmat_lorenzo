## POLIASTRO MAIN FEATURES

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


R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value

# intial parameters of RHW
a    = 6800 * u.kilometer
ecc  = .0016628 * u.one
inc  = 97.4864 * u.deg
raan = 39.164 * u.deg
argp = 325.3203 * u.deg
nu   = 126.202 * u.deg

epoch = Time("2018-11-30 03:53:03.550", scale="utc")
rhw0 = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

# parameters of RHW body
C_D = 2.2
A_over_m = ((np.pi/4.0) * (u.m**2) / (100 * u.kg)).to_value(u.km**2 / u.kg) # km**2/kg
B = C_D * A_over_m   # ballistic coefficient

# parameters of the atmosphere
rho0 = rho0_earth.to(u.kg/u.km**3).value
H0   = H0_earth.to(u.km).value


# #def pert(t0, u_, k):
#     du_kep = func_twobody(t0, u_, k)
#     ax, ay, az = J2_perturbation(
#         t0, u_, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
#     )
#     du_ad = np.array([0, 0, 0, ax, ay, az])
#     return du_kep + du_ad

#rhw1 = rhw0.propagate(10 * u.day, method = CowellPropagator(f=pert))


from numba import njit as jit
@jit
def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, A_over_m, H0, rho0
    )

tofs = TimeDelta(np.linspace(0, 10 * u.day, num=10 * 500))

def f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = a_d(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m,
        H0=H0,
        rho0=rho0,
        J2=Earth.J2.value,
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

rr3, _ = rhw0.to_ephem(EpochsArray(rhw0.epoch + tofs, method=CowellPropagator(f=f)),).rv()

# plot
fig, (axes1, axes2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 6))

axes1.plot(tofs.value, norm(rr3, axis=1) - Earth.R)
axes1.set_ylabel("h(t)")
axes1.set_xlabel("t, days")
axes1.set_ylim([400, 450])

plt.show()