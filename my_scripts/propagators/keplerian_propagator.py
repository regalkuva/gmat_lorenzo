## KEPLERIAN TWO-BODY PROPAGATOR (NO PERTUBATIONS)

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import Time
from poliastro.plotting import OrbitPlotter2D

import pandas as pd

import matplotlib.pyplot as plt

print('\nKEPLERIAN TWO-BODY PROPAGATOR\n')
# initial orbital elements
# RHW:
a    = 6865.501217 * u.km
ecc  = 0.0016628   * u.one
inc  = 97.4864     * u.deg
raan = 39.164      * u.deg
argp = 325.3203    * u.deg
nu   = 126.202     * u.deg 

start_date = Time("2018-11-30 03:53:03.550", scale = "utc")   # epoch

# Definition of the initial orbit (poliastro)
in_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, start_date)

# Keplerian two-body propagation (poliastro)
time_frame = float(input('Time frame [days]: ')) * u.day
time_step  = float(input('Time step [sec]: ')) * u.s

t = time_step
old_orbit = in_orbit

a_list     = [in_orbit.a.value]
ecc_list   = [in_orbit.ecc.value]
inc_list   = [in_orbit.inc.value]
raan_list  = [in_orbit.raan.value]
argp_list  = [in_orbit.argp.value]
nu_list    = [in_orbit.nu.value]
epoch_list = [in_orbit.epoch.value]

while t <= time_frame:
    
    new_orbit = old_orbit.propagate(time_step)   # two-body propagation
    
    a_list.append(new_orbit.a.value)
    ecc_list.append(new_orbit.ecc.value)
    inc_list.append(new_orbit.inc.to_value(u.deg))
    raan_list.append(new_orbit.raan.to_value(u.deg))
    argp_list.append(new_orbit.argp.to_value(u.deg))
    nu_list.append(new_orbit.nu.to_value(u.deg))
    epoch_list.append(new_orbit.epoch.value)

    old_orbit = new_orbit
    t = t + time_step

# orbital elements plot
fig, ax = plt.subplots(2,3, figsize=(20,12))
ax[0,0].plot(range(len(a_list)), a_list)
ax[0,0].set(title = "Semi-major axis vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "SMA [Km]")

ax[0,1].plot(range(len(a_list)), ecc_list)
ax[0,1].set(title = "Eccentricity vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "ECC")

ax[0,2].plot(range(len(a_list)), inc_list)
ax[0,2].set(title = "Inclination vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "INC [deg]")

ax[1,0].plot(range(len(a_list)), raan_list)
ax[1,0].set(title = "Right ascension of the ascending node vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "RAAN [deg]")

ax[1,1].plot(range(len(a_list)), argp_list)
ax[1,1].set(title = "Argument of the perigee vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "ARGP [deg]")

ax[1,2].plot(range(len(nu_list)), nu_list)
ax[1,2].set(title = "True anomaly vs Elapsed Time",
       xlabel = "t [sec]",
       ylabel = "TA [deg]")

plt.show()

# # 3d plot (IT NEEDS JUPYTER)
# op = OrbitPlotter2D()
# op.plot(orbit_in)

# Orbital data
data_table = zip(epoch_list, a_list, ecc_list, inc_list, raan_list, argp_list, nu_list)
df = pd.DataFrame(data = data_table, columns= ['Epoch [UTC]', 'SMA [Km]', 'ECC', 'INC [deg]', 'RAAN [deg]', 'ARGP [deg]', 'TA [deg]'])

print(df)

path = r'C:\Users\Lorenzo\Documents\GitHub\gmat_lorenzo\my_scripts\keplerian_propagator\rhw_1week.txt'

with open(path, 'a') as f:
    df_string = df.to_string()
    f.write(df_string)



