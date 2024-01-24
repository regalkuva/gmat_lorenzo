## ORBIT PROPAGATOR WITH J2 + ATMOSPHERIC DRAG PERTURBATIONS

from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.bodies import Earth
from poliastro.constants import rho0_earth, H0_earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag_exponential
from poliastro.core.propagation import func_twobody 
from poliastro.util import Time
#from poliastro.plotting import OrbitPlotter2D

from numba import njit as jit

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import time


print('\n--- TWO-BODY PROPAGATOR - J2 & ATMOSPHERIC DRAG PERTURBATIONS ---\n')
# initial orbital elements
# RHW:
a    = 6865.501217 * u.km         #6865.501217
ecc  = 0.0016628   * u.one
inc  = 97.4864     * u.deg
raan = 39.164      * u.deg
argp = 325.3203    * u.deg
nu   = 126.202     * u.deg 

start_date = Time("2018-11-30 03:53:03.550", scale = "utc")   # epoch 2018 11-30

# Definition of the initial orbit (poliastro)
in_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)

# Propagation time selection (poliastro)
# time_frame = float(input('- Time frame [days]: ')) * u.day
# time_step  = float(input('- Time step   [sec]: ')) * u.s

time_frame = 10<<u.day
time_step  = 3600<<u.s

process_start_time = time.time()   # start time of python code


# Constants
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value

rho0 = rho0_earth.to(u.kg/u.km**3).value
H0   = H0_earth.to(u.km).value

# GMAT correct RHW orbit decay was for: C_D = 2.2, A = 0.02 m^2, m = 2.205 kg
C_D = 2.2
A_over_m = ((0.01 * u.m**2) / (2.5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
B = C_D * A_over_m   # ballistic coefficient at low drag mode

a_list     = []
ecc_list   = []
inc_list   = []
raan_list  = []
argp_list  = []
nu_list    = []
epoch_list = []

@jit
def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):

        return J2_perturbation(
        t0, state, k, J2, R
        ) + atmospheric_drag_exponential(
            t0, state, k, R, C_D, A_over_m, H0, rho0
        )
        
def f(t0, state, k):
        
        du_kep = func_twobody(t0, state, k)
        ax, ay, az = a_d(
            t0, 
            state, 
            k=k, 
            J2 = J2, 
            R = R, 
            C_D = C_D, 
            A_over_m = A_over_m, 
            H0 = H0, 
            rho0 = rho0
            )
        du_ad = np.array([0, 0, 0, ax, ay, az])

        return du_kep + du_ad

secs = 0
elapsedsecs = []
numero = int(time_frame.to_value(u.s) / time_step.value)

tofs = TimeDelta(np.linspace(0, time_frame, num=numero))
ephem_tofs = in_orbit.to_ephem(EpochsArray(start_date + tofs, method=CowellPropagator(rtol=1e-5, f=f)))

for epoch in range(len(tofs)):
    
    secs += time_step.value
     
    orb_from_eph = Orbit.from_ephem(Earth, ephem_tofs, ephem_tofs.epochs[epoch])

    a_list.append(orb_from_eph.a.value)
    ecc_list.append(orb_from_eph.ecc.value)
    inc_list.append(orb_from_eph.inc.to_value(u.deg))
    raan_list.append(orb_from_eph.raan.to_value(u.deg))
    argp_list.append(orb_from_eph.argp.to_value(u.deg))
    nu_list.append(orb_from_eph.nu.to_value(u.deg))
    epoch_list.append(orb_from_eph.epoch.value)

    elapsedsecs.append(secs)

# Orbital elements data
# data_table = zip(epoch_list, a_list, ecc_list, inc_list, raan_list, argp_list, nu_list)
# df = pd.DataFrame(data = data_table, columns = [
#     'Epoch [UTC]', 'SMA [Km]', 'ECC', 'INC [deg]', 'RAAN [deg]', 'ARGP [deg]', 'TA [deg]'
#     ])
# print(df)

# Data to .txt file
# path = r'C:\Users\Lorenzo\Documents\GitHub\gmat_lorenzo\my_scripts\keplerian_propagator\rhw_1week.txt'

# with open(path, 'a') as f:
#     df_string = df.to_string()
#     f.write(df_string)

elapsed_days = []
for sec in range(len(elapsedsecs)):
    elapsed_days.append(elapsedsecs[sec]/(60*60*24))

print(f'\nProcess finished --- {time.time() - process_start_time}')

fig, ax = plt.subplots(2,3, figsize=(20,12))
ax[0,0].plot(elapsed_days, a_list)
ax[0,0].set(title = "Semi-major axis vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "SMA [Km]")

ax[0,1].plot(elapsed_days, ecc_list)
ax[0,1].set(title = "Eccentricity vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "ECC")

ax[0,2].plot(elapsed_days, inc_list)
ax[0,2].set(title = "Inclination vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "INC [deg]")

ax[1,0].plot(elapsed_days, raan_list)
ax[1,0].set(title = "Right ascension of the ascending node vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "RAAN [deg]")

ax[1,1].plot(elapsed_days, argp_list)
ax[1,1].set(title = "Argument of the perigee vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "ARGP [deg]")

ax[1,2].plot(elapsed_days, nu_list)
ax[1,2].set(title = "True anomaly vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "TA [deg]")

plt.show()