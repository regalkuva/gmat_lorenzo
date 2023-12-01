# ATMOSPHERIC DRAG FUNCTIONS

import numpy as np

from pyatmos import expo, jb2008

# Exponential model
def atmoexpo_pertubation(state, R, C_D, A_over_m):
    '''Provides atmospheric drag pertubating acceleration [km/s^2] 
    following an exponential model of the atmosphere.

    Inputs: - state:    Spacecraft state vector [x,y,z,v_x,v_y,v_z],  [km, km/s]
            - R:        Earth radius [km]
            - C_D:      Spacecraft aerodynamic drag coefficient
            - A_over_m: Spacecraft frontal area - mass ratio [km^2/kg]
    '''
    
    H = np.linalg.norm(state[:3])

    v_vec = state[3:]
    v = np.linalg.norm(v_vec)
    
    expo_geom = expo(H - R)
    rho = expo_geom.rho

    return - 0.5 * rho * C_D * A_over_m * v * v_vec

# JB2008 model
def jb2008_pertubation(epoch, state, R, C_D, A_over_m, swdata):
    '''Provides atmospheric drag pertubating acceleration [km/s^2] 
    following JB2008 model of the atmosphere.\n
    It needs space data weather:\n
    Download or update the space weather file from www.celestrak.com\n
    --- swfile = download_sw_nrlmsise00()\n
    Read the space weather data\n
    --- swdata = read_sw_nrlmsise00(swfile)\n\n

    # Inputs 
    - t0: str\n     
    \tTime(UTC), ex.: '2014-07-22 22:18:45'\n
    - state: numpy.ndarray\n    
    \tSpacecraft state vector [x,y,z,v_x,v_y,v_z],  [km, km/s]\n
    - C_D: float\n      
    \tSpacecraft aerodynamic drag coefficient\n
    - A_over_m: float\n 
    \tSpacecraft frontal area - mass ratio [km^2/kg]\n
    - swdata:   Space weather
    '''
    H = np.linalg.norm(state[:3])
    v_vec = state[3:]
    v = np.linalg.norm(v_vec)
     
    lat = np.degrees(np.arcsin(state[2]/H))
    lon = np.degrees(np.arctan2(state[1], state[0]))

    date = epoch

    jb08 = jb2008(date,(lat,lon,H-R),swdata)
    rho = jb08.rho

    return - 0.5 * rho * C_D * A_over_m * v * v_vec