from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import M_earth, GM_earth, J2_earth
from astropy.coordinates import Longitude, Angle, get_sun
from astropy import units as u

import math
import numpy as np
from numba import njit as jit

h = 500     #Input sat altitude
e = 1e-6    #Input sat eccentri

def inc_from_alt(h,e):
    J2 = J2_earth #Zonal harmonic constant
    r_earth = Earth.R_mean.value / 1000 #Earth radius

    µ_earth = GM_earth.value / 1e9
    r_orbit = r_earth + h #SMA

    rate = 0.19910213e-6 #Required precession rate rad/s

    v = math.sqrt(µ_earth/r_orbit) #m/s2
    p = (2 * math.pi * r_orbit / v) / 60
    i_rad = math.acos( (-2/3) * (rate) * (1/J2) * ( r_orbit * (1-e**2) / r_earth )**2 * math.sqrt( r_orbit**3 / µ_earth )  )
    i_deg = i_rad * (180/math.pi)


    return i_deg,p

def raan_from_ltan(time, ltan_in_hours):
    sun = get_sun(time)
    return Longitude(sun.ra + Angle(f'{ltan_in_hours - 12.0}h'))

def angle_between(v1: np.array,v2:np.array): 
    '''
    State vector for trailing satellite comes first. 
    Calculates angle between satellite in (0-360) range using atan2d
    '''
    # v1 = np.array([x2_list[i],y2_list[i],z2_list[i]])
    # v2 = np.array([x1_list[i],y1_list[i],z1_list[i]])
    n = np.array([1,0,0])
    n_norm = np.sqrt(sum(n**2))
    v1_proj = v1 - (np.dot(v1,n)/n_norm**2)*n
    v2_proj = v2 - (np.dot(v2,n)/n_norm**2)*n

    x = np.cross(v1_proj,v2_proj)
    c = np.sign(np.dot(x,n)) * np.linalg.norm(x)
    angle = np.rad2deg(np.arctan2(c,np.dot(v1,v2))) 
    angle = (angle+360)%360

    return angle


if __name__ == '__main__':
    inc_from_alt(h,e)
    print(f'Orbit radius is {Earth.R.value/1000 + h:.2f} km')
    print(f'Orbit altitude is {h:.2f} km')
    print(f'Orbital period is {inc_from_alt(h,e)[1]:.2f}  mins')
    print(f'Inclination required is {inc_from_alt(h,e)[0]:.4f} deg')


@jit
def argl_difference(ref_argp, ref_nu, trail_argp, trail_nu):
 
    ref_argp = ref_argp * 180/np.pi
    ref_nu = ref_nu * 180/np.pi
    trail_argp = trail_argp * 180/np.pi
    trail_nu = trail_nu * 180/np.pi



    argl_1 = (ref_argp + ref_nu)%360
    argl_2 = (trail_argp + trail_nu)%360

    #argl_1 = (reference_orbit.argp.to_value(u.deg) + reference_orbit.nu.to_value(u.deg))%360
    #argl_2 = (trailing_orbit.argp.to_value(u.deg) + trailing_orbit.nu.to_value(u.deg))%360
    
    return (argl_1 - argl_2)%360









