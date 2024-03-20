from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy import coordinates as coord
# from astropy.coordinates import EarthLocation, WGS84GeodeticRepresentation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator
from poliastro.util import Time, time_range
from poliastro.plotting import OrbitPlotter2D
from poliastro.earth import EarthSatellite, Spacecraft
from poliastro.earth.plotting import GroundtrackPlotter

from geopy.distance import geodesic
from haversine import haversine

import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go
import matplotlib.pyplot as plt

# import sys
# sys.path.append('../sso_functions')
# from inc_from_smaecc import required_inc


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

# sensor parameters
sw = 100*0.5 # [km]

# 25 km --> 0.22483 deg of latitude

targets_coord = {'Naples': (40.8518, 14.2681), 'Helsinki': (60.1699, 24.9384)}

target = targets_coord['Naples']   # latitude/longitude [deg]

# Definition of the initial orbit (poliastro)
in_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)

# Spacecraft data
C_D = 2.2  * u.one
A   = 0.01 * u.m**2
m   = 2.5  * u.kg

rhw_sc = Spacecraft(A, C_D, m)

# Keplerian two-body propagation (poliastro)
time_frame = 5 * u.s   #float(input('Time frame [days]: '))
time_step  = 5 * u.s     #float(input('Time step [sec]: '))

number = int(time_frame.to_value(u.s) / time_step.value)
tofs = TimeDelta(np.linspace(0, time_frame, num=number))
t_span = time_range(start=start_date, periods=number, end=start_date + time_frame)



ephems = in_orbit.to_ephem(EpochsArray(start_date + tofs, CowellPropagator(rtol=1e-5)))

# xyz_0  = in_orbit.represent_as(coord.CartesianRepresentation)
# gcrs_0 = coord.GCRS(xyz_0, obstime=start_date)
# itrs_0 = gcrs_0.transform_to(coord.ITRS(obstime=start_date))
# loc_0  = coord.EarthLocation.from_geocentric(itrs_0.x, itrs_0.y, itrs_0.z)
# lon_0, lat_0, _ = loc_0.to_geodetic()
# target = (lat_0.value, lon_0.value)


access_time = []
for i in range(len(tofs)):
    curr_orbit = Orbit.from_ephem(Earth, ephems, ephems.epochs[i])

    xyz  = curr_orbit.represent_as(coord.CartesianRepresentation)
    gcrs = coord.GCRS(xyz, obstime=ephems.epochs[i])
    itrs = gcrs.transform_to(coord.ITRS(obstime=ephems.epochs[i]))
    loc  = coord.EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
     
    lon, lat, _ = loc.to_geodetic()
    pos = (lat.value, lon.value)
    #dist = haversine(pos, target)
    dist = geodesic(pos, target).km
    if  dist < sw:
        access_time.append(ephems.epochs[i])

print(f'\nAccess times = {access_time}\n')
print(f'\nProcess finished --- {int(time.time() - process_start_time)}')