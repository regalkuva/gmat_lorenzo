from astropy import units as u
from astropy.time import TimeDelta
from astropy import coordinates as coord

from poliastro.util import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.propagation import func_twobody
from poliastro.core.perturbations import J2_perturbation

import sys
sys.path.append('../my_scripts')

from sso_functions.inc_from_smaecc import required_inc
from propagators.perturbations import coesa76_model

from haversine import haversine

import numpy as np
import time


process_start_time = time.time()   # start time (instant) of python code
start_date_time = time.ctime()  # start time (clock) of python code
print(f'\n--- WALKER DELTA CONSTELLATION --- [process start: {start_date_time}]\n')

# Constants
R = (Earth.R).to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value

# Orbital inputs
alt = 380 * u.km
a   = R*u.km + alt
ecc = 0.001 * u.one
inc = required_inc(a.value, ecc.value) * u.deg
argp = 0 * u.deg

start_date = Time("2023-01-01 12:00:00.000", scale = "utc")

# Satellite parameters
C_D = 2.2
A_over_m = ((0.01 * u.m**2) / (2.5 * u.kg)).to_value(u.km**2 / u.kg)   # km**2/kg

# sensor parameters
sw = 50 * 0.5 # [km]

# target coordinates
targets_coord = {'Naples': (40.8518, 14.2681), 'Helsinki': (60.1699, 24.9384)}
target = targets_coord['Naples']   # latitude/longitude [deg]

# Propagation time parameters
time_frame = 1 * u.day   #float(input('Time frame [days]: ')) * u.day
time_step  = 8 * u.s #float(input('Time step [sec]: ')) * u.s

number = int(time_frame.to_value(u.s) / time_step.value)
tofs = TimeDelta(np.linspace(0, time_frame, num=number))

# Walker Delta pattern --> inc:t/p/f
t = 90     # total number of satellites
p = 3      # orbital planes
f = 0.75   # phasing parameter
delta_nu = f * 360 / t 
s = int(t/p)   # satellites per orbit

# Perturbations
def a_d(t0, state, k, J2, R, C_D, A_over_m):

        return J2_perturbation(
        t0, state, k, J2, R
        ) + coesa76_model(
            state, R, C_D, A_over_m
        )
        
def f_pert(t0, state, k):
        
        du_kep = func_twobody(t0, state, k)
        ax, ay, az = a_d(
            t0, 
            state, 
            k=k, 
            J2 = J2, 
            R = R, 
            C_D = C_D, 
            A_over_m = A_over_m
            )
        du_ad = np.array([0, 0, 0, ax, ay, az])

        return du_kep + du_ad

sats_orbit_list = []
for plane in range(p):
    raan = (plane * 360 /p) * u.deg

    for sat in range(s):
        nu = (sat*360/s + delta_nu*plane) * u.deg
        orbit_0 = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)
        ephem = orbit_0.to_ephem(EpochsArray(start_date + tofs, method=CowellPropagator(rtol=1e-5, f=f_pert)))
        sats_orbit_list.append(ephem)

access_time = []
for inst in range(len(tofs)):

    for sat in range(t):
        curr_orbit = Orbit.from_ephem(Earth, sats_orbit_list[sat], sats_orbit_list[0].epochs[inst])
        xyz  = curr_orbit.represent_as(coord.CartesianRepresentation)
        gcrs = coord.GCRS(xyz, obstime=sats_orbit_list[0].epochs[inst])
        itrs = gcrs.transform_to(coord.ITRS(obstime=sats_orbit_list[0].epochs[inst]))
        loc  = coord.EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)

        lon, lat, _ = loc.to_geodetic()
        pos = (lat.value, lon.value)
        dist = haversine(pos, target)

        if  dist < sw:
            access_time.append(sats_orbit_list[0].epochs[inst])
                

print(f'\nAccess Times = {access_time}\n')
print(f'\nProcess finished --- {int(time.time() - process_start_time)}')