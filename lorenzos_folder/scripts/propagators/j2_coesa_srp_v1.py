## ORBIT PROPAGATOR WITH J2 + ATMOSPHERIC DRAG PERTURBATIONS

from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.ephem import build_ephem_interpolant
from poliastro.constants import Wdivc_sun, rotational_period_sun
from poliastro.core.perturbations import J2_perturbation, radiation_pressure
from poliastro.core.propagation import func_twobody 
from poliastro.util import Time
#from poliastro.plotting import OrbitPlotter2D

from pyatmos import coesa76

import numpy as np

import matplotlib.pyplot as plt

import time
from datetime import datetime

import sys
sys.path.append('../scripts')

from sso_functions.orbital_elements import osc2mean
from sso_functions.sso_inc import inc_from_alt, raan_from_ltan


print('\n--- TWO-BODY PROPAGATOR - J2 & ATMOSPHERIC DRAG PERTURBATIONS ---\n')

# Propagation time selection (poliastro)
# time_frame = float(input('- Time frame [days]: ')) * u.day
# time_step  = float(input('- Time step   [sec]: ')) * u.s
time_frame = 1<<u.day
time_step  = 8.64<<u.s
start_date = datetime(2025,1,1,12,0,0)
epoch = Time(val=start_date.isoformat(), format='isot')
start_date_ephem = Time("2025-01-01 12:00:00.000", scale = "utc")

# Constants
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value
W_over_c = Wdivc_sun.value
r_sun = build_ephem_interpolant(Sun, rotational_period_sun, (start_date_ephem, start_date_ephem + time_frame), rtol=1e-5)


# initial orbital elements
h = 380
ltan = 22.5

a = (R + h) << u.km
ecc = 1e-6 << u.one
inc = inc_from_alt(h,ecc)[0] << u.deg   
raan = raan_from_ltan(Time(val=datetime.timestamp(start_date), format='unix'),ltan) << u.deg
argp = 1e-6 << u.deg
nu = 1e-6 << u.deg

# Definition of the initial orbit (poliastro)
in_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)



process_start_time = time.time()   # start time of python code

# Satellite
C_D = 2.2
A_over_m = ((0.2 * u.m**2) / (50 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg
B = C_D * A_over_m   # ballistic coefficient at low drag mode
C_R = 1.8

a_list     = []
ecc_list   = []
inc_list   = []
raan_list  = []
argp_list  = []
nu_list    = []
epoch_list = []

a_mean_list = []
ecc_mean_list = []
inc_mean_list = []
raan_mean_list = []
argp_mean_list = []
ma_mean_list = []


def coesa76_model(state, R, C_D, A_over_m):
    H = np.linalg.norm(state[:3])

    v_vec = state[3:]
    v = np.linalg.norm(v_vec)
    
    coesa_geom = coesa76(H - R, 'geometric')
    rho = (coesa_geom.rho)
    rho = rho*1e9  #(u.kg/u.m**3)).to_value(u.kg/u.km**3)

    return - 0.5 * rho * C_D * A_over_m * v * v_vec


def j2_coesa_srp(t0, state, k, J2, R, C_D, A_over_m, C_R, W_over_c, r_star):
    return J2_perturbation(
        t0, state, k, J2, R
        ) + coesa76_model(
            state, R, C_D, A_over_m
            ) + radiation_pressure(
                t0, state, k, R, C_R, A_over_m, W_over_c, r_star)

    
def f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = j2_coesa_srp(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m,
        J2=J2,
        C_R=C_R,
        W_over_c=W_over_c,
        star=r_sun
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad


number = int(time_frame.to_value(u.s) / time_step.value)
tofs = TimeDelta(np.linspace(0, time_frame, num=number))

ephem_tofs = in_orbit.to_ephem(EpochsArray(start_date_ephem + tofs, method=CowellPropagator(rtol=1e-5, f=f)))

secs = 0
elapsedsecs = []

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
    
    mean_elements = osc2mean(
          orb_from_eph.a.value,
          orb_from_eph.ecc.value,
          orb_from_eph.inc.value,
          orb_from_eph.raan.value,
          orb_from_eph.argp.value,
          orb_from_eph.nu.value
    )

    a_mean_list.append(mean_elements[0])
    ecc_mean_list.append(mean_elements[1])
    inc_mean_list.append(mean_elements[2])
    raan_mean_list.append(mean_elements[3])
    argp_mean_list.append(mean_elements[4])
    ma_mean_list.append(mean_elements[5])

    elapsedsecs.append(secs)

elapsed_days = []
for sec in range(len(elapsedsecs)):
    elapsed_days.append(elapsedsecs[sec]/(60*60*24))

altitudes = []
mean_altitudes = []
for sma in range(len(a_list)):
    altitudes.append(a_list[sma] - Earth.R_mean.to_value(u.km))
    mean_altitudes.append(a_mean_list[sma] - Earth.R_mean.to_value(u.km))

print(f'\nProcess finished --- {time.time() - process_start_time}')

fig, ax = plt.subplots(2, 3, figsize=(22,9), squeeze=False) 

ax[0,0].plot(elapsed_days, altitudes, label='Osculating Altitude')
ax[0,0].plot(elapsed_days, mean_altitudes, label='Mean Altitude')
ax[0,0].legend(loc = 'center right')
ax[0,0].set_title('Altitude')

ax[0,1].plot(elapsed_days, ecc_list, label='Osculating ECC')
ax[0,1].plot(elapsed_days, ecc_mean_list, label='Mean ECC')
ax[0,1].legend(loc = 'center right')
ax[0,1].set_title('ECC')

ax[1,0].plot(elapsed_days, inc_list, label='Osculating INC')
ax[1,0].plot(elapsed_days, inc_mean_list, label='Mean INC')
ax[1,0].legend(loc = 'center right')
ax[1,0].set_title('INC')

ax[1,1].plot(elapsed_days, raan_list, label='Osculating RAAN')
ax[1,1].plot(elapsed_days, raan_mean_list, label='Mean RAAN')
ax[1,1].legend(loc = 'upper left')
ax[1,1].set_title('RAAN')

ax[0,2].plot(elapsed_days, argp_list, label='Osculating ARGP')
ax[0,2].plot(elapsed_days, argp_mean_list, label='Mean ARGP')
ax[0,2].set_title('ARGP')

ax[1,2].plot(elapsed_days, nu_list, label='Osculating TA')
ax[1,2].plot(elapsed_days, ma_mean_list, label='Mean MA')
ax[1,2].set_title('MEAN ANOMALY')

plt.show()