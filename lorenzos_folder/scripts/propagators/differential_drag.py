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

# initial orbital elements of CHASER and TARGET
# Constellation parameters
T = 2

# Assumptions: same spacecrafts, same orbital plane
# CHASER (1):
a_1, ecc_1, inc_1, raan_1, argp_1, nu_1 = 6995 * u.km, 0.001 * u.one, 97.98 * u.deg, 69 * u.deg, 275 * u.deg, 80 * u.deg

# TARGET (2):
a_2, ecc_2, inc_2, raan_2, argp_2, nu_2 = 6990 * u.km, 0.001 * u.one, 97.98 * u.deg, 69 * u.deg, 275 * u.deg, 160 * u.deg 

# Let's assign the same start date to the satellites
start_date = Time("2014-04-17 13:12:43.89", scale = "utc")

# definition of initial orbits by using poliastro
in_orbit_1 = Orbit.from_classical(Earth, a_1, ecc_1, inc_1, raan_1, argp_1, nu_1, start_date)
in_orbit_2 = Orbit.from_classical(Earth, a_2, ecc_2, inc_2, raan_2, argp_2, nu_2, start_date)

# Constants
R  = Earth.R.to(u.km).value
k  = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value

rho0 = rho0_earth.to(u.kg/u.km**3).value
H0   = H0_earth.to(u.km).value

# Only two attitude modes: high drag (HD) and low drag (LD)
C_D = 2.2
LD_A_over_m = ((0.02 * u.m**2) / (5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
HD_A_over_m = ((0.08 * u.m**2) / (5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
LD_B = C_D * LD_A_over_m   # ballistic coefficient at low drag mode
HD_B = C_D * LD_A_over_m   # ballistic coefficient at high drag mode

# Let's propagate the two satellite to compute the mean semi_major axis values
# functions to get the pertubating effects:
@jit
def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    return J2_perturbation(
        t0, state, k, J2, R
        ) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, A_over_m, H0, rho0
    )

def f(t0, state, k):

    du_kep = func_twobody(t0, state, k)
    ax, ay, az = a_d(t0, 
                     state, 
                     k, 
                     J2 = J2, 
                     R = R, 
                     C_D = C_D, 
                     A_over_m = LD_A_over_m, 
                     H0 = H0, 
                     rho0 = rho0)
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

# propagation for 1 day
# def j2_drag_prop(time_step, time_frame, initial_orbit, )
# a_list_c     = [in_orbit_c.a.value]
# ecc_list_c   = [in_orbit_c.ecc.value]
# inc_list_c   = [in_orbit_c.inc.value]
# raan_list_c  = [in_orbit_c.raan.value]
# argp_list_c  = [in_orbit_c.argp.value]
# nu_list_c    = [in_orbit_c.nu.value]
# epoch_list_c = [in_orbit_c.epoch.value]

# time_step  = 1 * u.s
# time_frame = 1 * u.day 

# t = time_step
# old_orbit = in_orbit

# while t <= time_frame:

#     new_orbit = old_orbit.propagate(time_step, method=CowellPropagator(f=f))

#     a_list.append(new_orbit.a.value)
#     ecc_list.append(new_orbit.ecc.value)
#     inc_list.append(new_orbit.inc.to_value(u.deg))
#     raan_list.append(new_orbit.raan.to_value(u.deg))
#     argp_list.append(new_orbit.argp.to_value(u.deg))
#     nu_list.append(new_orbit.nu.to_value(u.deg))
#     epoch_list.append(new_orbit.epoch.value)

#     old_orbit = new_orbit
#     t = t + time_step


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

# Accelerations provided by poliastro:

@jit
def a_d(t0, state, k, R, C_D, A_over_m, H0, rho0):
    return atmospheric_drag_exponential(t0, state, k, R, C_D, A_over_m, H0, rho0)

# acceleration in high drag mode
HD_acc = a_d(t0=0, state = array_2, k=k, R=R, C_D=C_D, A_over_m=HD_A_over_m, H0=H0, rho0=rho0)

# acceleration in low drag mode
LD_acc = a_d(t0=0, state = array_2, k=k, R=R, C_D=C_D, A_over_m=LD_A_over_m, H0=H0, rho0=rho0)


unit_HD_acc = HD_acc / np.linalg.norm(HD_acc)
unit_LD_acc = LD_acc / np.linalg.norm(LD_acc)

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


