import numpy as np

import matplotlib.pyplot as plt

import time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import norm
from poliastro.constants import R_earth
from poliastro.core.elements import coe2rv
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator

from astropy import units as u
from astropy.time import Time, TimeDelta

from datetime import datetime, timedelta
from sso_inc import inc_from_alt, raan_from_ltan,angle_between

from perturbations import pertubations_coesa_high, pertubations_coesa_low

from osc2mean_dd import osc2mean


toc = time.time()
## Orbit
h = 300
start_date = datetime(2024,1,1,9,0,0)
ltan = 22.5

a = (R_earth.value/1000 + h) << u.km
ecc = 1e-6 << u.one
inc = inc_from_alt(h,ecc)[0] << u.deg   
raan = raan_from_ltan(Time(val=datetime.timestamp(start_date), format='unix'),ltan) << u.deg
argp = 1e-6 << u.deg
nu = 1e-6 << u.deg

epoch = Time(val=start_date.isoformat(), format='isot')

delta_a = 2
delta_nu = -10

reference_orbit = Orbit.from_classical(
    Earth,
    a,
    ecc,
    inc,
    raan,
    argp,
    nu,
    epoch
    )
trailing_orbit = Orbit.from_classical(
    Earth,
    (a.value+delta_a)<<u.km,
    ecc,
    inc,
    raan,
    argp,
    (nu.value+delta_nu)<<u.deg,
    epoch
    )


time_step = 3600<<u.s

refsmalist = []
refsmalist_mean = []
trailsmalist = []
trailsmalist_mean = []

assignment = 10

ref_vel = []
trail_vel = []

elapsedsecs = []
secs = 0

rmag_ref = []
rmag_trail = []

vmag_ref = []
vmag_trail = []

angle_list = []
ang_vel_list = []

theta_err_list = []

maneuvers = 0

#for timestamp in range(len(timestamps)):
start_date_prop = epoch
ref_mean = osc2mean(a.value, ecc.value, inc.value, raan.value, argp.value, nu.value)
ref_mean_orbit = Orbit.from_classical(Earth, ref_mean[0]<<u.km, ref_mean[1]<<u.one, ref_mean[2]<<u.deg, ref_mean[3]<<u.deg, ref_mean,[4]<<u.deg, ref_mean[5]<<u.deg, epoch)
trail_mean = osc2mean(a.value+delta_a, ecc, inc.value, raan.value, argp.value, nu.value+delta_nu)
trail_mean_orbit = Orbit.from_classical(Earth, trail_mean[0]<<u.km, trail_mean[1]<<u.one, trail_mean[2]<<u.deg, trail_mean[3]<<u.deg, trail_mean,[4]<<u.deg, trail_mean[5]<<u.deg, epoch)



while trail_mean[0] > ref_mean[0]: 
    
    # theta_err = 10 - reference_orbit.nu.to_value(u.deg)
    theta_err = 10 - angle_between(trail_mean_orbit.r.value, ref_mean_orbit.r.value)
    #theta_err = reference_orbit.nu.to_value(u.deg) - trailing_orbit.nu.to_value(u.deg) + 10
    tra_orb_10days = trail_mean_orbit.propagate(10<<u.day, method=CowellPropagator(rtol=1e-5, f=pertubations_coesa_high))
    theta_dot_dot = (tra_orb_10days.n.to_value(u.deg/u.s) - trail_mean_orbit.n.to_value(u.deg/u.s)) / (10*24*60*60)
    t_hd = (ref_mean_orbit.n.to_value(u.deg/u.s) - trail_mean_orbit.n.to_value(u.deg/u.s)) / theta_dot_dot
    theta_hd = 0.5 * theta_dot_dot * t_hd**2
    t_wait = np.abs(theta_hd - theta_err) / (ref_mean_orbit.n.to_value(u.deg/u.s) - trail_mean_orbit.n.to_value(u.deg/u.s))

        

    num_wait = int(t_wait / time_step.value)
    tofs_wait = TimeDelta(np.linspace(0, t_wait<<u.s, num=num_wait))
    reference_ephem = reference_orbit.to_ephem(EpochsArray(start_date_prop + tofs_wait, method=CowellPropagator(rtol=1e-5, f=pertubations_coesa_low)))
    trailing_ephem = trailing_orbit.to_ephem(EpochsArray(start_date_prop + tofs_wait, method=CowellPropagator(rtol=1e-5, f=pertubations_coesa_low)))

    for t in range(len(tofs_wait)):

        secs += time_step.value

        refsmalist.append(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).a.value)
        trailsmalist.append(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).a.value)

        # ref_mean = osc2mean(
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).a.value,
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).ecc.value,
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).inc.value,
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).raan.value,
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).argp.value,
        #     Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).nu.value

        # )
        # trail_mean = osc2mean(
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).a.value,
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).ecc.value,
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).inc.value,
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).raan.value,
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).argp.value,
        #     Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).nu.value
        # )
        # refsmalist_mean.append(ref_mean[0])
        # trailsmalist_mean.append(trail_mean[0])

        rmag_ref.append(np.linalg.norm(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).r.value))
        rmag_trail.append(np.linalg.norm(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).r.value))  

        vmag_ref.append(np.linalg.norm(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).v.value))
        vmag_trail.append(np.linalg.norm(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).v.value))
        
        angle_list.append(angle_between(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).r.value, Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).r.value))
        # angle_list.append((Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).nu.to_value(u.deg)) - Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).nu.value)

        ang_vel_ref = (360 << u.deg) / Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).period
        ang_vel_trail = (360 <<u.deg) / Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).period
        ang_vel_diff =  ang_vel_ref - ang_vel_trail
        ang_vel_list.append(ang_vel_diff.value)

        elapsedsecs.append(secs)
    
    reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
    trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])

    num_hd = int(t_hd / time_step.value)
    tofs_hd = TimeDelta(np.linspace(0, t_hd<<u.s, num=num_hd))

    reference_ephem = reference_orbit.to_ephem(EpochsArray(reference_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=pertubations_coesa_low)))
    trailing_ephem = trailing_orbit.to_ephem(EpochsArray(trailing_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=pertubations_coesa_high)))

    for t in range(len(tofs_hd)):

        secs += time_step.value

        refsmalist.append(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).a.value)
        trailsmalist.append(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).a.value)

        rmag_ref.append(np.linalg.norm(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).r.value))
        rmag_trail.append(np.linalg.norm(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).r.value))  

        vmag_ref.append(np.linalg.norm(Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).v.value))
        vmag_trail.append(np.linalg.norm(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).v.value))
        
        angle_list.append(angle_between(Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).r.value, Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).r.value))
        # angle_list.append((Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).nu.to_value(u.deg)) - Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).nu.value)
        
        ang_vel_ref = (360 << u.deg) / Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).period
        ang_vel_trail = (360 <<u.deg) / Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).period
        ang_vel_diff =  ang_vel_ref - ang_vel_trail
        ang_vel_list.append(ang_vel_diff.value)

        elapsedsecs.append(secs)

    reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
    trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])

    ref_mean = osc2mean(reference_orbit.a.value, reference_orbit.ecc.value, reference_orbit.inc.value, reference_orbit.raan.value, reference_orbit.argp.value, reference_orbit.nu.value)
    ref_mean_orbit = Orbit.from_classical(Earth, ref_mean[0]<<u.km, ref_mean[1]<<u.one, ref_mean[2]<<u.deg, ref_mean[3]<<u.deg, ref_mean,[4]<<u.deg, ref_mean[5]<<u.deg, reference_orbit.epoch)

    trail_mean = osc2mean(trailing_orbit.a.value, ecc, trailing_orbit.inc.value, trailing_orbit.raan.value, trailing_orbit.argp.value, trailing_orbit.nu)
    trail_mean_orbit = Orbit.from_classical(Earth, trail_mean[0]<<u.km, trail_mean[1]<<u.one, trail_mean[2]<<u.deg, trail_mean[3]<<u.deg, trail_mean,[4]<<u.deg, trail_mean[5]<<u.deg, trailing_orbit.epoch)

    start_date_prop = reference_orbit.epoch


fig, ax = plt.subplots(2, 3, figsize=(22,9), squeeze=False) 

ax[0,0].plot(elapsedsecs,trailsmalist,label='Trail')
ax[0,0].plot(elapsedsecs,refsmalist,label='Ref')
ax[0,0].legend(loc = 'center right')
ax[0,0].set_title('Ref vs Trail SMA')

ax[0,1].plot(elapsedsecs,rmag_trail,label='Trail')
ax[0,1].plot(elapsedsecs,rmag_ref,label='Ref')
ax[0,1].legend(loc = 'center right')
ax[0,1].set_title('Ref vs Trail RMAG')

ax[1,0].plot(elapsedsecs,vmag_trail,label='Trail')
ax[1,0].plot(elapsedsecs,vmag_ref, label='Ref')
ax[1,0].legend(loc = 'center right')
ax[1,0].set_title('Ref vs Trail VMAG')

# z = np.polyfit(elapsedsecs, angle_list, 1)
# p = np.poly1d(z)

ax[1,1].plot(elapsedsecs,angle_list)
# ax[1,1].plot(elapsedsecs,p(elapsedsecs))
ax[1,1].axhline(assignment,linestyle='--',color='red',label = f'Assigned Slot at {assignment}deg')
ax[1,1].legend(loc = 'upper left')
ax[1,1].set_title('Angle Between Satellites')

ax[0,2].plot(elapsedsecs,ang_vel_list)
ax[0,2].set_title('Angular Vel. Difference between Satellites')

# plt.show(block=True)

tic = time.time()
print(f'Timestep {time_step:.4f}s')
print(f'Run time {tic-toc:.2f}s/{(tic-toc)/60:.2f}m')
plt.show()