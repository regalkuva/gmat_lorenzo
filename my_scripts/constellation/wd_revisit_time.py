## Revisit time calculator for a Walker Delta constellation under Keplerian assumptions

from astropy import units as u
from astropy.time import TimeDelta
from astropy import coordinates as coord

from poliastro.util import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator

from haversine import haversine

from inc_from_smaecc import required_inc

import numpy as np
import time

from poliastro.plotting import OrbitPlotter3D
from matplotlib import pyplot as plt

# 3 planes, 90 satellite, wd: i:90/3/5, 380km SSO 

process_start_time = time.time()   # start time (instant) of python code
start_date_time = time.ctime()  # start time (clock) of python code
print(f'\n--- REVISIT TIME FOR A WALKER DELTA CONSTELLATION --- [process start: {start_date_time}]\n')

R = (Earth.R).to(u.km)

alt = 380 * u.km
a   = R + alt
ecc = 0.001 * u.one
inc = required_inc(a.value, ecc.value) * u.deg
argp = 0 * u.deg

start_date = Time("2023-01-01 00:00:00.000", scale = "utc")

time_frame = 5 * u.day   #float(input('Time frame [days]: ')) * u.day
time_step  = 1 * u.s #float(input('Time step [sec]: ')) * u.s

number = int(time_frame.to_value(u.s) / time_step.value)
tofs = TimeDelta(np.linspace(0, time_frame, num=number))

# sensor parameters
sw = 50 * 0.5 # [km]

targets_coord = {'Naples': (40.8518, 14.2681), 'Helsinki': (60.1699, 24.9384)}
target = targets_coord['Naples']   # latitude/longitude [deg]

# Walker Delta pattern --> inc:t/p/f
t = 90     # total number of satellites
p = 3      # orbital planes
f = 0.75   # phasing parameter
delta_nu = f * 360 / t 
s = int(t/p)   # satellites per orbit

sats_orbit_list = []
for plane in range(p):
    raan = (plane * 360 /p) * u.deg

    for sat in range(s):
        nu = (sat*360/s + delta_nu*plane) * u.deg
        orbit_0 = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)
        ephem = orbit_0.to_ephem(EpochsArray(start_date + tofs, method=CowellPropagator(rtol=1e-5)))
        sats_orbit_list.append(ephem)


access_time = []
revisit_time = []
j = 0

while len(revisit_time) == 0:

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
            print(dist)

            if  dist < sw:
                access_time.append(sats_orbit_list[0].epochs[inst])
                
                if (len(access_time)>1) and (((access_time[j].jd - access_time[j-1].jd)) > ((5*u.s).to_value(u.day))):
                    revisit_time.append(access_time[j].jd - access_time[j-1].jd)

                j += 1


print(f'\nRevisit time = {revisit_time} s\n')
print(f'\nProcess finished --- {int(time.time() - process_start_time)}')


# # plot
# frame = OrbitPlotter3D()

# for sat in range(t):
#     orbita_0 = Orbit.from_ephem(Earth, sats_orbit_list[sat], sats_orbit_list[0].epochs[0])
#     frame.plot(orbita_0)
    
# plt.savefig( "Two-Body Trajectory (3D View) for .png" )
# plt.show()