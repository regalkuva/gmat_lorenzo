## --- 2 SATELLITES DIFFERENTIAL DRAG ALGORITHM ---

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.constants import rho0_earth, H0_earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag_exponential
from poliastro.core.propagation import func_twobody 
from poliastro.util import Time

from numba import njit as jit

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print('DIFFERENTIAL DRAG CONTROL - 2 SATELLITES')

## 1 - INITIALIZATION

# initial orbital elements of FLOCK 1C-1 (40027) and FLOCK 1C-2 (40029) from TLEs
# Constellation parameters
T = 2

# FLOCK 1C-1:
a1, ecc1, inc1, raan1, argp1, nu1 = 6991.261179 * u.km, 0.0011698 * u.one, 97.9876 * u.deg, 69.189 * u.deg, 278.7152 * u.deg, 81.2675 * u.deg 
n1 = 14.85147101   # mean motion [revolutions per day]

# FLOCK 1C-2:
a2, ecc2, inc2, raan2, argp2, nu2 = 6992.126263 * u.km, 0.0012992 * u.one, 97.988 * u.deg, 69.0532 * u.deg, 275.1305 * u.deg, 84.8392 * u.deg 
n2 = 14.8487149   # mean motion [revolutions per day]

# Let's suppose the same start date (1C-1 has been selected, they differ by around 4 hours)
start_date = Time("2014-04-17 13:12:43.89", scale = "utc")

# definition of initial orbits by using poliastro
in_orbit_1 = Orbit.from_classical(Earth, a1, ecc1, inc1, raan1, argp1, nu1, start_date)
in_orbit_2 = Orbit.from_classical(Earth, a2, ecc2, inc2, raan2, argp2, nu2, start_date)

# Constants
R  = Earth.R.to(u.km).value
k  = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value

rho0 = rho0_earth.to(u.kg/u.km**3).value
H0   = H0_earth.to(u.km).value

C_D = 2.2
LD_A_over_m = ((np.pi/4.0) * (0.01 * u.m**2) / (5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
HD_A_over_m = ((np.pi/4.0) * (0.238 * u.m**2) / (5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
LD_B = C_D * LD_A_over_m   # ballistic coefficient at low drag mode
HD_B = C_D * LD_A_over_m   # ballistic coefficient at high drag mode

# Problem vectors
in_orbit_1_state = in_orbit_1.rv()
in_orbit_2_state = in_orbit_2.rv()


unit_in_1_r = in_orbit_1_state[0] / np.linalg.norm(in_orbit_1_state[0])
unit_in_1_v = in_orbit_1_state[1] / np.linalg.norm(in_orbit_1_state[1])
unit_in_2_r = in_orbit_2_state[0] / np.linalg.norm(in_orbit_2_state[0])
unit_in_2_v = in_orbit_2_state[1] / np.linalg.norm(in_orbit_2_state[1])

dot_product_r = np.dot(unit_in_2_r, unit_in_1_r)
dot_product_v = np.dot(unit_in_2_v, unit_in_1_v)

theta     = np.arccos(dot_product_r).to_value(u.deg)   # angular difference between sats
theta_dot = np.arccos(dot_product_v)                   # angular velocity difference

array_2 = np.array([*in_orbit_2_state[0].value, *in_orbit_2_state[1].value])

@jit
def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, A_over_m, H0, rho0
    )
# acceleration in high drag mode
HD_a_d = a_d(t0=0, state = array_2, k=k, J2=J2, R=R, C_D=C_D, A_over_m=HD_A_over_m, H0=H0, rho0=rho0)

# acceleration in low drag mode
LD_a_d = a_d(t0=0, state = array_2, k=k, J2=J2, R=R, C_D=C_D, A_over_m=LD_A_over_m, H0=H0, rho0=rho0)

unit_HD_acc = HD_a_d / np.linalg.norm(HD_a_d)
unit_LD_acc = LD_a_d / np.linalg.norm(LD_a_d)

dot_product_acc = np.dot(unit_HD_acc, unit_LD_acc)
theta_dot_dot   = np.arccos(dot_product_acc)

## 2 - DIFFERENTIAL DRAG ALGORITHM




## ORBIT DETERMINATION
# time preferences selection
time_frame = 1 * u.day
time_step  = 1 * u.s

@jit
def LD_a_d(t0, state, k, J2, R, C_D, LD_A_over_m, H0, rho0):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, LD_A_over_m, H0, rho0
    )

@jit
def HD_a_d(t0, state, k, J2, R, C_D, HD_A_over_m, H0, rho0):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, HD_A_over_m, H0, rho0
    )


