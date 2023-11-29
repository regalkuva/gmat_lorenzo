from math import acos

from astropy import units as u

from poliastro.bodies import Earth

def required_inc(sma, ecc):
    '''This functions provides the required inclination related to the 
     semi-major axis (sma) and eccentricity (ecc) values of a Sun Synchronous Orbit (SSO).
     Inputs:
     - sma [km]
     - ecc'''

    alpha_sun_dot = .98 * (u.deg / u.day)
    alpha_sun_dot = alpha_sun_dot.to(u.rad / u.s).value

    J2 = Earth.J2.value
    R = Earth.R.to(u.km).value
    k = Earth.k.to(u.km**3 / u.s**2).value
    
    p = (sma * (1 - ecc**2))   # semi-latus rectum [km]
    n = ((k / sma**3) ** (1/2)) # mean motion [rad/s]

    inc = (acos(-(2/3) * (alpha_sun_dot/J2) * ((p/R)**2) / n)) * u.rad
    inc = inc.to(u.deg).value

    return inc

# # main
R = Earth.R.to(u.km).value
h = 380   #altitude
sma = R + h
ecc = .00001
inc = required_inc(sma, ecc)
print(f'The required inclination is {inc} degrees')