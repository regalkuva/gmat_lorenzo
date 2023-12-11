from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy import coordinates as coord

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator
from poliastro.util import Time, time_range

import numpy as np
import pandas as pd
import time


process_start_time = time.time()   # start time of python code
start_date_time = time.ctime()  # start time of python code
print(f'\nREVISIT TIME FOR A SPECIFIC LOCATION ON EARTH --- [process start: {start_date_time}]\n')

# initial orbital elements
alt  = 380 * u.km
a    = (Earth.R).to(u.km) + alt
ecc  = 0.001 * u.one
inc  = 96.9172647751491 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu   = 0 * u.deg 

start_date = Time("2023-01-01 12:00:00.000", scale = "utc")   # epoch

in_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)

# sensor parameters
#sw = 400*0.5 # [km]

targets_coord = {'Naples': (40.8518, 14.2681), 'Helsinki': (60.1699, 24.9384)}
target = targets_coord['Naples']

# 25 km --> 0.22483 deg of latitude
sw_lat = 0.22483 * 4
sw_lon = 0.4500 * 4   # 25 km of sw for helsinki's latitude

lat_range = pd.Interval(target[0] - sw_lat, target[0] + sw_lat)
lon_range = pd.Interval(target[-1] - sw_lon, target[-1] + sw_lon)


# Keplerian two-body propagation (poliastro)
time_frame = 6.1 * u.day   #float(input('Time frame [days]: '))
time_step  = 5 * u.s     #float(input('Time step [sec]: '))

number = int(time_frame.to_value(u.s) / time_step.value)
tofs = TimeDelta(np.linspace(0, time_frame, num=number))
t_span = time_range(start=start_date, periods=number, end=start_date + time_frame)

ephems = in_orbit.to_ephem(EpochsArray(start_date + tofs, CowellPropagator(rtol=1e-5)))

revisit_time = []
i = 0

while len(revisit_time) < 3:
    curr_orbit = Orbit.from_ephem(Earth, ephems, ephems.epochs[i])

    xyz  = curr_orbit.represent_as(coord.CartesianRepresentation)
    gcrs = coord.GCRS(xyz, obstime=ephems.epochs[i])
    itrs = gcrs.transform_to(coord.ITRS(obstime=ephems.epochs[i]))
    loc  = coord.EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
     
    lon, lat, _ = loc.to_geodetic()
    pos = (lat.value, lon.value)

    if pos[0] in lat_range and pos[-1] in lon_range:
            revisit_time.append(ephems.epochs[i])
    
    i += 1

print(revisit_time)

    





