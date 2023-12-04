## ORBIT PROPAGATOR WITH J2 + ATMOSPHERIC DRAG PERTUBATIONS

from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Earth
from poliastro.constants import rho0_earth, H0_earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody 
from poliastro.util import Time
#from poliastro.plotting import OrbitPlotter2D

from atmo_drag_functions import jb2008_pertubation

#from numba import njit as jit

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import time


print('\n--- TWO-BODY PROPAGATOR - J2 & ATMOSPHERIC DRAG PERTUBATIONS ---\n')
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
time_frame = float(input('- Time frame [days]: ')) * u.day
time_step  = float(input('- Time step   [sec]: ')) * u.s

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

# JB200 atmpospheric pertubation
from pyatmos import download_sw_jb2008,read_sw_jb2008
# Download or update the space weather file from https://sol.spacenvironment.net
swfile = download_sw_jb2008() 
# Read the space weather data
swdata = read_sw_jb2008(swfile) 

a_list     = [in_orbit.a.value]
ecc_list   = [in_orbit.ecc.value]
inc_list   = [in_orbit.inc.value]
raan_list  = [in_orbit.raan.value]
argp_list  = [in_orbit.argp.value]
nu_list    = [in_orbit.nu.value]
epoch_list = [in_orbit.epoch.value]

t = time_step
old_orbit = in_orbit

def a_d(t0, state, k, J2, R, C_D, A_over_m, epoch, swdata):

        return J2_perturbation(
        t0, state, k, J2, R
        ) + jb2008_pertubation(
            epoch, state, R, C_D, A_over_m, swdata
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
            epoch=date, 
            swdata=swdata
            )
        du_ad = np.array([0, 0, 0, ax, ay, az])

        return du_kep + du_ad



while t <= time_frame:
    
    date = old_orbit.epoch.value

    new_orbit = old_orbit.propagate(time_step, method=CowellPropagator(rtol=1e-5, f=f))


    a_list.append(new_orbit.a.value)
    ecc_list.append(new_orbit.ecc.value)
    inc_list.append(new_orbit.inc.to_value(u.deg))
    raan_list.append(new_orbit.raan.to_value(u.deg))
    argp_list.append(new_orbit.argp.to_value(u.deg))
    nu_list.append(new_orbit.nu.to_value(u.deg))
    epoch_list.append(new_orbit.epoch.value)

    old_orbit = new_orbit
    t = t + time_step

# Orbital elements data
data_table = zip(epoch_list, a_list, ecc_list, inc_list, raan_list, argp_list, nu_list)
df = pd.DataFrame(data = data_table, columns = [
    'Epoch [UTC]', 'SMA [Km]', 'ECC', 'INC [deg]', 'RAAN [deg]', 'ARGP [deg]', 'TA [deg]'
    ])
print(df)

# Data to .txt file
# path = r'C:\Users\Lorenzo\Documents\GitHub\gmat_lorenzo\my_scripts\keplerian_propagator\rhw_1week.txt'

# with open(path, 'a') as f:
#     df_string = df.to_string()
#     f.write(df_string)

# RANDOM UPDATE FOR GITHUB
print(f'\nProcess finished --- {time.time() - process_start_time}')

fig, ax = plt.subplots(2,3, figsize=(20,12))
ax[0,0].plot(range(len(a_list)), a_list)
ax[0,0].set(title = "Semi-major axis vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "SMA [Km]")

ax[0,1].plot(range(len(a_list)), ecc_list)
ax[0,1].set(title = "Eccentricity vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "ECC")

ax[0,2].plot(range(len(a_list)), inc_list)
ax[0,2].set(title = "Inclination vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "INC [deg]")

ax[1,0].plot(range(len(a_list)), raan_list)
ax[1,0].set(title = "Right ascension of the ascending node vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "RAAN [deg]")

ax[1,1].plot(range(len(a_list)), argp_list)
ax[1,1].set(title = "Argument of the perigee vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "ARGP [deg]")

ax[1,2].plot(range(len(a_list)), nu_list)
ax[1,2].set(title = "True anomaly vs Elapsed Time",
       xlabel = "t [days]",
       ylabel = "TA [deg]")

plt.show()