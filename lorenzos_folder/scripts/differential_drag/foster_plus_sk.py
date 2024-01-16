import numpy as np

import matplotlib.pyplot as plt

import time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import norm
from poliastro.constants import R_earth
from poliastro.core.elements import coe2rv
from poliastro.core.propagation import func_twobody 
from poliastro.twobody.thrust import change_a_inc
from poliastro.twobody.sampling import EpochsArray
from poliastro.twobody.propagation import CowellPropagator

from astropy import units as u
from astropy.time import Time, TimeDelta

from datetime import datetime
from sso_inc import inc_from_alt, raan_from_ltan,angle_between

from perturbations import perturbations_coesa_J2_low, perturbations_coesa_J2_high, coesa_J2
from osc2mean_dd import osc2mean


toc = time.time()

## Orbit
h = 350
start_date = datetime(2024,1,1,9,0,0)
ltan = 22.5

a = (R_earth.value/1000 + h) << u.km
ecc = 1e-6 << u.one
inc = inc_from_alt(h,ecc)[0] << u.deg   
raan = raan_from_ltan(Time(val=datetime.timestamp(start_date), format='unix'),ltan) << u.deg
argp = 1e-6 << u.deg
nu = 1e-6 << u.deg

epoch = Time(val=start_date.isoformat(), format='isot')

delta_a = 1
delta_nu = 0

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
assignment = 50%360
pred_days = 20

refsmalist = []
refsmalist_mean = []
trailsmalist = []
trailsmalist_mean = []



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
mean_ang_list = []
hd_window = []
hd_duration = []

start_date_prop = epoch
ref_mean = osc2mean(a.value, ecc.value, inc.value, raan.value, argp.value, nu.value)
ref_mean_orbit = Orbit.from_classical(Earth, ref_mean[0]<<u.km, ref_mean[1]<<u.one, ref_mean[2]<<u.deg, ref_mean[3]<<u.deg, ref_mean[4]<<u.deg, nu, epoch)
trail_mean = osc2mean(a.value+delta_a, ecc.value, inc.value, raan.value, argp.value, nu.value+delta_nu)
trail_mean_orbit = Orbit.from_classical(Earth, trail_mean[0]<<u.km, trail_mean[1]<<u.one, trail_mean[2]<<u.deg, trail_mean[3]<<u.deg, trail_mean[4]<<u.deg, nu+(delta_nu<<u.deg), epoch)

mans = 2
## Foster Algorithm (Commissioning Phase)
for i in range(mans):

    theta_err = (assignment - angle_between(trailing_orbit.r.value, reference_orbit.r.value))%360
    
    tra_orb_pred = trailing_orbit.propagate(pred_days<<u.day, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_high))
    tra_pred_mean = osc2mean(tra_orb_pred.a.value, tra_orb_pred.ecc.value, tra_orb_pred.inc.to_value(u.deg), tra_orb_pred.raan.to_value(u.deg), tra_orb_pred.argp.to_value(u.deg), tra_orb_pred.nu.to_value(u.deg))
    tra_orb_pred_mean = Orbit.from_classical(Earth, tra_pred_mean[0]<<u.km, tra_pred_mean[1]<<u.one, tra_pred_mean[2]<<u.deg, tra_pred_mean[3]<<u.deg, tra_pred_mean[4]<<u.deg, tra_orb_pred.nu.to(u.deg), tra_orb_pred.epoch)
    
    theta_dot_dot = (tra_orb_pred_mean.n.to(u.deg/u.s) - trail_mean_orbit.n.to(u.deg/u.s)) / ((pred_days*60*60*24)<<u.s)
    t_hd = (ref_mean_orbit.n.to(u.deg/u.s) - trail_mean_orbit.n.to(u.deg/u.s)) / theta_dot_dot
    theta_hd = 0.5 * theta_dot_dot * t_hd**2
    t_wait = (theta_err - theta_hd.value) / (ref_mean_orbit.n.to_value(u.deg/u.s) - trail_mean_orbit.n.to_value(u.deg/u.s))       

    num_wait = int(t_wait / time_step.value)
    tofs_wait = TimeDelta(np.linspace(0, t_wait<<u.s, num=num_wait))
    reference_ephem = reference_orbit.to_ephem(EpochsArray(start_date_prop + tofs_wait, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))
    trailing_ephem = trailing_orbit.to_ephem(EpochsArray(start_date_prop + tofs_wait, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))

    hd_window.append(secs+t_wait)
    hd_duration.append(t_hd)


    for t in range(len(tofs_wait)):

        secs += time_step.value

        ref_from_ephem = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t])
        trail_from_ephem = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t])

        refsmalist.append(ref_from_ephem.a.value)
        trailsmalist.append(trail_from_ephem.a.value)

        ref_mean = osc2mean(
            ref_from_ephem.a.value,
            ref_from_ephem.ecc.value,
            ref_from_ephem.inc.to_value(u.deg),
            ref_from_ephem.raan.to_value(u.deg),
            ref_from_ephem.argp.to_value(u.deg),
            ref_from_ephem.nu.to_value(u.deg)

        )
        trail_mean = osc2mean(
            trail_from_ephem.a.value,
            trail_from_ephem.ecc.value,
            trail_from_ephem.inc.to_value(u.deg),
            trail_from_ephem.raan.to_value(u.deg),
            trail_from_ephem.argp.to_value(u.deg),
            trail_from_ephem.nu.to_value(u.deg)
        )

        refsmalist_mean.append(ref_mean[0])
        trailsmalist_mean.append(trail_mean[0])

        rmag_ref.append(np.linalg.norm(ref_from_ephem.r.value))
        rmag_trail.append(np.linalg.norm(trail_from_ephem.r.value))  

        vmag_ref.append(np.linalg.norm(ref_from_ephem.v.value))
        vmag_trail.append(np.linalg.norm(trail_from_ephem.v.value))
        
        angle_list.append(angle_between(trail_from_ephem.r.value, ref_from_ephem.r.value))

        ang_vel_ref = (360 << u.deg) / ref_from_ephem.period
        ang_vel_trail = (360 <<u.deg) / trail_from_ephem.period
        ang_vel_diff =  ang_vel_ref - ang_vel_trail
        ang_vel_list.append(ang_vel_diff.value)

        elapsedsecs.append(secs)
    
    reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
    trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])


    num_hd = int(t_hd.value / time_step.value)
    tofs_hd = TimeDelta(np.linspace(0, t_hd, num=num_hd))

    reference_ephem = reference_orbit.to_ephem(EpochsArray(reference_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))
    trailing_ephem = trailing_orbit.to_ephem(EpochsArray(trailing_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_high)))


    for t in range(len(tofs_hd)):

        secs += time_step.value

        ref_from_ephem = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t])
        trail_from_ephem = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t])

        refsmalist.append(ref_from_ephem.a.value)
        trailsmalist.append(trail_from_ephem.a.value)

        ref_mean = osc2mean(
            ref_from_ephem.a.value,
            ref_from_ephem.ecc.value,
            ref_from_ephem.inc.to_value(u.deg),
            ref_from_ephem.raan.to_value(u.deg),
            ref_from_ephem.argp.to_value(u.deg),
            ref_from_ephem.nu.to_value(u.deg)

        )
        trail_mean = osc2mean(
            trail_from_ephem.a.value,
            trail_from_ephem.ecc.value,
            trail_from_ephem.inc.to_value(u.deg),
            trail_from_ephem.raan.to_value(u.deg),
            trail_from_ephem.argp.to_value(u.deg),
            trail_from_ephem.nu.to_value(u.deg)
        )

        refsmalist_mean.append(ref_mean[0])
        trailsmalist_mean.append(trail_mean[0])

        if ref_mean[0] == trail_mean[0]:
            print(f'Same altitude achieved at: {secs}[s]')

        rmag_ref.append(np.linalg.norm(ref_from_ephem.r.value))
        rmag_trail.append(np.linalg.norm(trail_from_ephem.r.value))  

        vmag_ref.append(np.linalg.norm(ref_from_ephem.v.value))
        vmag_trail.append(np.linalg.norm(trail_from_ephem.v.value))
        
        angle_list.append(angle_between(trail_from_ephem.r.value, ref_from_ephem.r.value))
        #angle_list.append((Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).nu.to_value(u.deg)) - Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).nu.value)

        ang_vel_ref = (360 << u.deg) / ref_from_ephem.period
        ang_vel_trail = (360 <<u.deg) / trail_from_ephem.period
        ang_vel_diff =  ang_vel_ref - ang_vel_trail
        ang_vel_list.append(ang_vel_diff.value)

        elapsedsecs.append(secs)

    reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
    trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])

    ref_mean = osc2mean(
        reference_orbit.a.value, 
        reference_orbit.ecc.value, 
        reference_orbit.inc.to_value(u.deg), 
        reference_orbit.raan.to_value(u.deg), 
        reference_orbit.argp.to_value(u.deg), 
        reference_orbit.nu.to_value(u.deg)
        )
    ref_mean_orbit = Orbit.from_classical(
                                          Earth, 
                                          ref_mean[0]<<u.km, 
                                          ref_mean[1]<<u.one, 
                                          ref_mean[2]<<u.deg, 
                                          ref_mean[3]<<u.deg, 
                                          ref_mean[4]<<u.deg, 
                                          reference_orbit.nu.to(u.deg), 
                                          reference_orbit.epoch
                                          )

    trail_mean = osc2mean(
        trailing_orbit.a.value, 
        trailing_orbit.ecc.value, 
        trailing_orbit.inc.to_value(u.deg), 
        trailing_orbit.raan.to_value(u.deg), 
        trailing_orbit.argp.to_value(u.deg), 
        trailing_orbit.nu.to_value(u.deg)
        )
    trail_mean_orbit = Orbit.from_classical(
                                            Earth, 
                                            trail_mean[0]<<u.km, 
                                            trail_mean[1]<<u.one, 
                                            trail_mean[2]<<u.deg, 
                                            trail_mean[3]<<u.deg, 
                                            trail_mean[4]<<u.deg, 
                                            trailing_orbit.nu.to(u.deg), 
                                            trailing_orbit.epoch
                                            )

    start_date_prop = reference_orbit.epoch
    #pred_days = pred_days * 0.5

## Semi-Major Axis Difference Nullifying (Commissioning Phase)
pred_days = 1
sma_mans = 1

for i in range(sma_mans):

    tra_orb_pred = trailing_orbit.propagate(pred_days<<u.day, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_high))
    tra_pred_mean = osc2mean(tra_orb_pred.a.value, tra_orb_pred.ecc.value, tra_orb_pred.inc.to_value(u.deg), tra_orb_pred.raan.to_value(u.deg), tra_orb_pred.argp.to_value(u.deg), tra_orb_pred.nu.to_value(u.deg))
    tra_orb_pred_mean = Orbit.from_classical(Earth, tra_pred_mean[0]<<u.km, tra_pred_mean[1]<<u.one, tra_pred_mean[2]<<u.deg, tra_pred_mean[3]<<u.deg, tra_pred_mean[4]<<u.deg, tra_orb_pred.nu.to(u.deg), tra_orb_pred.epoch)

    theta_dot_dot = (tra_orb_pred_mean.n.to(u.deg/u.s) - trail_mean_orbit.n.to(u.deg/u.s)) / ((pred_days*60*60*24)<<u.s)
    t_hd = (ref_mean_orbit.n.to(u.deg/u.s) - trail_mean_orbit.n.to(u.deg/u.s)) / theta_dot_dot
    #t_hd = (0.7*60*60*24)<<u.s
    num_hd = int(t_hd.value / time_step.value)
    tofs_hd = TimeDelta(np.linspace(0, t_hd, num=num_hd))

    reference_ephem = reference_orbit.to_ephem(EpochsArray(reference_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))
    trailing_ephem = trailing_orbit.to_ephem(EpochsArray(trailing_ephem.epochs[-1] + tofs_hd, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_high)))

    for t in range(len(tofs_hd)):

        secs += time_step.value

        ref_from_ephem = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t])
        trail_from_ephem = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t])

        refsmalist.append(ref_from_ephem.a.value)
        trailsmalist.append(trail_from_ephem.a.value)

        ref_mean = osc2mean(
            ref_from_ephem.a.value,
            ref_from_ephem.ecc.value,
            ref_from_ephem.inc.to_value(u.deg),
            ref_from_ephem.raan.to_value(u.deg),
            ref_from_ephem.argp.to_value(u.deg),
            ref_from_ephem.nu.to_value(u.deg)

        )
        trail_mean = osc2mean(
            trail_from_ephem.a.value,
            trail_from_ephem.ecc.value,
            trail_from_ephem.inc.to_value(u.deg),
            trail_from_ephem.raan.to_value(u.deg),
            trail_from_ephem.argp.to_value(u.deg),
            trail_from_ephem.nu.to_value(u.deg)
        )

        refsmalist_mean.append(ref_mean[0])
        trailsmalist_mean.append(trail_mean[0])

        if ref_mean[0] == trail_mean[0]:
            print(f'Same altitude achieved at: {secs}[s]')

        rmag_ref.append(np.linalg.norm(ref_from_ephem.r.value))
        rmag_trail.append(np.linalg.norm(trail_from_ephem.r.value))  

        vmag_ref.append(np.linalg.norm(ref_from_ephem.v.value))
        vmag_trail.append(np.linalg.norm(trail_from_ephem.v.value))
        
        angle_list.append(angle_between(trail_from_ephem.r.value, ref_from_ephem.r.value))
        #angle_list.append((Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t]).nu.to_value(u.deg)) - Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t]).nu.value)

        ang_vel_ref = (360 << u.deg) / ref_from_ephem.period
        ang_vel_trail = (360 <<u.deg) / trail_from_ephem.period
        ang_vel_diff =  ang_vel_ref - ang_vel_trail
        ang_vel_list.append(ang_vel_diff.value)

        elapsedsecs.append(secs)


# Propagation without control
# t_prop = (60*60*24*30)<<u.s
# num_prop = int(t_prop.value / time_step.value)
# tofs_prop = TimeDelta(np.linspace(0, t_prop, num=num_prop))

# reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
# trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])

# reference_ephem = reference_orbit.to_ephem(EpochsArray(reference_ephem.epochs[-1] + tofs_prop, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))
# trailing_ephem = trailing_orbit.to_ephem(EpochsArray(trailing_ephem.epochs[-1] + tofs_prop, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low)))

# for t in range(len(tofs_prop)):

#     secs += time_step.value

#     ref_from_ephem = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[t])
#     trail_from_ephem = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[t])

#     refsmalist.append(ref_from_ephem.a.value)
#     trailsmalist.append(trail_from_ephem.a.value)

#     ref_mean = osc2mean(
#         ref_from_ephem.a.value,
#         ref_from_ephem.ecc.value,
#         ref_from_ephem.inc.to_value(u.deg),
#         ref_from_ephem.raan.to_value(u.deg),
#         ref_from_ephem.argp.to_value(u.deg),
#         ref_from_ephem.nu.to_value(u.deg)

#     )
#     trail_mean = osc2mean(
#         trail_from_ephem.a.value,
#         trail_from_ephem.ecc.value,
#         trail_from_ephem.inc.to_value(u.deg),
#         trail_from_ephem.raan.to_value(u.deg),
#         trail_from_ephem.argp.to_value(u.deg),
#         trail_from_ephem.nu.to_value(u.deg)
#     )

#     refsmalist_mean.append(ref_mean[0])
#     trailsmalist_mean.append(trail_mean[0])

#     rmag_ref.append(np.linalg.norm(ref_from_ephem.r.value))
#     rmag_trail.append(np.linalg.norm(trail_from_ephem.r.value))  

#     vmag_ref.append(np.linalg.norm(ref_from_ephem.v.value))
#     vmag_trail.append(np.linalg.norm(trail_from_ephem.v.value))
    
#     angle_list.append(angle_between(trail_from_ephem.r.value, ref_from_ephem.r.value))

#     ang_vel_ref = (360 << u.deg) / ref_from_ephem.period
#     ang_vel_trail = (360 <<u.deg) / trail_from_ephem.period
#     ang_vel_diff =  ang_vel_ref - ang_vel_trail
#     ang_vel_list.append(ang_vel_diff.value)

#     elapsedsecs.append(secs)


## Sation-Keeping
reference_orbit = Orbit.from_ephem(Earth, reference_ephem, reference_ephem.epochs[-1])
trailing_orbit = Orbit.from_ephem(Earth, trailing_ephem, trailing_ephem.epochs[-1])

R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2)
k_val = k.value
R_mean = Earth.R_mean.to_value(u.km)
J2 = Earth.J2.value

C_D = 2.2
A_over_m = ((0.01 << u.m**2) / (100 * u.kg)).to_value(u.km**2 / u.kg)
acc = 2.4e-7 * (u.km / u.s**2)

a_down = (R_mean + 335)<<u.km
a_up   = (R_mean + 365)<<u.km
inc_up = inc_from_alt(350,ref_mean[1])[0] << u.deg

def f_thrust_trail(t0, state, k):

    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa_J2(
        t0, 
        state, 
        k=k_val,
        J2=J2, 
        R = R, 
        C_D = C_D, 
        A_over_m = A_over_m
        )
    
    ax_t, ay_t, az_t = a_d_thrust_trail(t0,
                                  state,
                                  k=k)
    du_ad = np.array([0, 0, 0, ax+ax_t, ay+ay_t, az+az_t])
    
    return du_kep + du_ad
    
def f_thrust_ref(t0, state, k):

    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa_J2(
        t0, 
        state, 
        k=k_val,
        J2=J2, 
        R = R, 
        C_D = C_D, 
        A_over_m = A_over_m
        )
    
    ax_t, ay_t, az_t = a_d_thrust_ref(t0,
                                  state,
                                  k=k)
    du_ad = np.array([0, 0, 0, ax+ax_t, ay+ay_t, az+az_t])

    return du_kep + du_ad


t_sk = (60*60*24*25)<<u.s
num_sk = int(t_sk.value / time_step.value)
tofs_sk = TimeDelta(np.linspace(0, t_sk, num=num_sk))

no_maneuver = True

for t in range(len(tofs_sk)):

    secs += time_step.value

    if  ref_mean[0] > a_down.value and no_maneuver:
        reference_orbit = reference_orbit.propagate(time_step, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low))
        trailing_orbit = trailing_orbit.propagate(time_step, method=CowellPropagator(rtol=1e-5, f=perturbations_coesa_J2_low))

    else:
        no_maneuver = False
        a_down_ref = reference_orbit.a
        a_down_trail = trailing_orbit.a
        inc_down_ref =  ref_mean[2] << u.deg
        inc_down_trail = trail_mean[2]  << u.deg

        a_d_thrust_ref, d_v, t_f = change_a_inc(k, a_down_ref, a_up, inc_down_ref, inc_up, acc)
        a_d_thrust_trail, d_v, t_f = change_a_inc(k, a_down_trail, a_up, inc_down_trail, inc_up, acc)

        reference_orbit = reference_orbit.propagate(time_step, method=CowellPropagator(rtol=1e-5, f=f_thrust_ref))
        trailing_orbit = trailing_orbit.propagate(time_step, method=CowellPropagator(rtol=1e-5, f=f_thrust_trail))


    refsmalist.append(reference_orbit.a.value)
    trailsmalist.append(trailing_orbit.a.value)

    ref_mean = osc2mean(
        reference_orbit.a.value,
        reference_orbit.ecc.value,
        reference_orbit.inc.to_value(u.deg),
        reference_orbit.raan.to_value(u.deg),
        reference_orbit.argp.to_value(u.deg),
        reference_orbit.nu.to_value(u.deg)

    )
    trail_mean = osc2mean(
        trailing_orbit.a.value,
        trailing_orbit.ecc.value,
        trailing_orbit.inc.to_value(u.deg),
        trailing_orbit.raan.to_value(u.deg),
        trailing_orbit.argp.to_value(u.deg),
        trailing_orbit.nu.to_value(u.deg)
    )

    if ref_mean[0] > a_up.value:
        no_maneuver = True

    refsmalist_mean.append(ref_mean[0])
    trailsmalist_mean.append(trail_mean[0])

    rmag_ref.append(np.linalg.norm(reference_orbit.r.value))
    rmag_trail.append(np.linalg.norm(trailing_orbit.r.value))  

    vmag_ref.append(np.linalg.norm(reference_orbit.v.value))
    vmag_trail.append(np.linalg.norm(trailing_orbit.v.value))
    
    angle_list.append(angle_between(trailing_orbit.r.value, reference_orbit.r.value))

    ang_vel_ref = (360 << u.deg) / reference_orbit.period
    ang_vel_trail = (360 <<u.deg) / trailing_orbit.period
    ang_vel_diff =  ang_vel_ref - ang_vel_trail
    ang_vel_list.append(ang_vel_diff.value)

    elapsedsecs.append(secs)




elapsed_days = []
for sec in range(len(elapsedsecs)):
    elapsed_days.append(elapsedsecs[sec]/(60*60*24))

trail_mean_altitudes = []
for sma in range(len(trailsmalist_mean)):
    trail_mean_altitudes.append(trailsmalist_mean[sma] - Earth.R_mean.to_value(u.km))

ref_mean_altitudes = []
for sma in range(len(refsmalist_mean)):
    ref_mean_altitudes.append(refsmalist_mean[sma] - Earth.R_mean.to_value(u.km))

fig, ax = plt.subplots(2, 3, figsize=(22,9), squeeze=False) 

ax[0,0].plot(elapsed_days,trailsmalist,label='Trail')
ax[0,0].plot(elapsed_days,refsmalist,label='Ref')
ax[0,0].legend(loc = 'center right')
ax[0,0].set_title('Ref vs Trail SMA')
ax[0,0].set_xlabel('Days')
ax[0,0].set_ylabel('Km')

ax[0,1].plot(elapsed_days,rmag_trail,label='Trail')
ax[0,1].plot(elapsed_days,rmag_ref,label='Ref')
ax[0,1].legend(loc = 'center right')
ax[0,1].set_title('Ref vs Trail RMAG')
ax[0,1].set_xlabel('Days')
ax[0,1].set_ylabel('Km')

ax[1,0].plot(elapsed_days,vmag_trail,label='Trail')
ax[1,0].plot(elapsed_days,vmag_ref, label='Ref')
ax[1,0].legend(loc = 'center right')
ax[1,0].set_title('Ref vs Trail VMAG')
ax[1,0].set_xlabel('Days')
ax[1,0].set_ylabel('Km/s')

ax[1,1].plot(elapsed_days,angle_list)
ax[1,1].axhline(assignment,linestyle='--',color='red',label = f'Assigned Slot at {assignment}deg')
ax[1,1].legend(loc = 'upper left')
ax[1,1].set_title('Angle Between Satellites')
ax[1,1].set_xlabel('Days')
ax[1,1].set_ylabel('Degrees')

ax[0,2].plot(elapsed_days,ang_vel_list)
ax[0,2].set_title('Angular Vel. Difference between Satellites')
ax[0,2].set_xlabel('Days')
ax[0,2].set_ylabel('Degrees/s')

ax[1,2].plot(elapsed_days,trail_mean_altitudes,label='Trail')
ax[1,2].plot(elapsed_days,ref_mean_altitudes,label='Ref')
ax[1,2].set_title('Ref vs Trail Mean Altitudes')
ax[1,2].set_xlabel('Days')
ax[1,2].set_ylabel('Km')


print(f'Starting HD windows time step [s]: {hd_window}')
print(f'HD windows duration [s]: {hd_duration}')

tic = time.time()
print(f'Timestep {time_step:.4f}s')
print(f'Run time {tic-toc:.2f}s/{(tic-toc)/60:.2f}m')
plt.show()