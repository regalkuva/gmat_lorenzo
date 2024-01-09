from astropy import units as u

from poliastro.constants import J2_earth, R_earth

import numpy as np

J2 = J2_earth.value
R  = R_earth.to_value(u.km)

def osc2mean(a, ecc, inc, raan, argp, nu):

    # a = a.value
    # inc = inc.to_value(u.rad)
    # raan = raan.to_value(u.rad)
    # argp = argp.to_value(u.rad)
    # nu = nu.to_value(u.rad)

    a = a
    inc = (inc<<u.deg).to_value(u.rad)
    raan = (raan<<u.deg).to_value(u.rad)
    argp = (argp<<u.deg).to_value(u.rad)
    nu = (nu<<u.deg).to_value(u.rad)

    ecc_anomaly = np.arccos((ecc + np.cos(nu))/(1 + ecc*np.cos(nu)))
    ma = ecc_anomaly - ecc*np.sin(ecc_anomaly)
    # supposing ecc<<1: ma = nu

    gamma_2 = - J2 * 0.5 * (R/a)**2   #(G.297)
    eta = (1 - ecc**2)**(0.5)
    gamma_2_mean = gamma_2 / eta**4   #(G.298)
    a_over_r = (1 + ecc*np.cos(nu)) / eta**2   # r: current orbit radius ; (G.301)

    #(G.302)
    a_mean = a + a * gamma_2 * ((3*(np.cos(inc)**2) - 1) * (a_over_r**3 - (1/eta**3)) + 3*(1-(np.cos(inc)**2))*(a_over_r**3)*np.cos(2*argp + 2*nu))

    #(G.303)
    delta_ecc_1 = (gamma_2_mean/8)*ecc*(eta**2) * (1 - 11*(np.cos(inc)**2) - 40*(np.cos(inc)**4)/(1-5*(np.cos(inc)**2))) * np.cos(2*argp)
    #(G.304)
    delta_ecc = delta_ecc_1 + (eta**2)*0.5*(
        gamma_2 * (((3*(np.cos(inc)**2) - 1)/(eta**6)) * (
            ecc*eta + (ecc/(1+eta)) + 3*np.cos(nu) + 3*ecc*(np.cos(nu)**2) + (ecc**2)*(np.cos(nu)**3) 
            ) + (3/(eta**6))*(1-(np.cos(inc)**2))*(ecc + 3*np.cos(nu) * 3*ecc*(np.cos(nu)**2) + (ecc**2)*(np.cos(nu)**3)
    )*np.cos(2*argp + 2*nu)
    ) - gamma_2_mean*(1-(np.cos(inc)**2))*(3*np.cos(2*argp + nu)) + np.cos(2*argp + 2*nu)
    )
    
    #(G.305)
    delta_inc = - (ecc*delta_ecc_1/((eta**2)*np.tan(inc))) + gamma_2_mean*0.5*np.cos(inc)*np.sqrt(1-(np.cos(inc)**2))*(
        3*np.cos(2*argp + 2*nu) + 3*ecc*np.cos(2*argp + nu) + ecc*np.cos(2*argp + 3*nu)
    )
    
    #(G.306)
    ma_argp_raan_mean = ma + argp + raan + (gamma_2_mean/8)*(eta**3)*(
        1 - 11*(np.cos(inc)**2) - 40*(np.cos(inc)**4)/(1-5*(np.cos(inc)**2))
    ) - (gamma_2_mean/16)*(
        2 + (ecc**2) - 11*(2+3*(ecc**2))*(np.cos(inc)**2) - 40*(2+5*(ecc**2))*(np.cos(inc)**4)/(1-5*(np.cos(inc)**2)) - 400*(ecc**2)*(np.cos(inc)**6)/((1-5*(np.cos(inc)**2))**2)
    ) + (gamma_2_mean/4)*(
        -6*(1-5*(np.cos(inc)**2))*(nu - ma + ecc*np.sin(nu)) + (3-5*(np.cos(inc)**2))*(3*np.sin(2*argp+2*nu) + 3*ecc*np.sin(2*argp+nu)) + ecc*np.sin(2*argp+3*nu)
    ) - (gamma_2_mean/8)*(ecc**2)*np.cos(inc)*(
        11 + 80*(np.cos(inc)**2)/(1-5*(np.cos(inc)**2)) + 200*(np.cos(inc)**4)/((1-5*(np.cos(inc)**2))**2)
    ) - (gamma_2_mean/2)*np.cos(inc)*(
        6*(nu-ma+ecc*np.sin(nu)) - 3*np.sin(2*argp+2*nu) - 3*ecc*np.sin(2*argp+nu) - ecc*np.sin(2*argp+3*nu)
    )

    #(G.307)
    ecc_x_delta_ma = (gamma_2_mean/8)*ecc*(eta**3)*(
        1 - 11*(np.cos(inc)**2) - 40*(np.cos(inc)**4)/(1-5*(np.cos(inc)**2))
        ) - (gamma_2_mean/4)*(eta**3)*(
            2*(3*(np.cos(inc)**2)-1)*((a*eta/R)**2+(a/R)+1)*np.sin(nu) + 3*(1-(np.cos(inc)**2))*(
                (((-a*eta/R)**2)-(a/R)+1)*np.sin(2*argp+nu) + ((a*eta/R)**2+(a/R)+(1/3))*np.sin(2*argp+3*nu)
            )
        )
    
    #(G.308)
    delta_raan = (-gamma_2_mean/8)*(ecc**2)*np.cos(inc)*(
        11 + 80*(np.cos(inc)**2)/(1-5*np.cos(inc)**2) + 200*(np.cos(inc)**4)/((1-5*np.cos(inc)**2)**2)
    ) - gamma_2_mean*0.5*np.cos(inc)*(
        6*(nu-ma+ecc*np.sin(nu)) - 3*np.sin(2*argp+2*nu) - 3*ecc*np.sin(2*argp+nu) - ecc*np.sin(2*argp+3*nu)
    )

    d_1 = (ecc+delta_ecc)*np.sin(ma) + ecc_x_delta_ma*np.cos(ma)   #(G.309)
    d_2 = (ecc+delta_ecc)*np.cos(ma) - ecc_x_delta_ma*np.sin(ma)   #(G.310)

    ma_mean = np.arctan(d_1/d_2)   #(G.311)

    ecc_mean = np.sqrt(d_1**2 + d_2**2)   #(G.312)

    d_3 = (np.sin(inc/2) + np.cos(inc/2)*delta_inc/2)*np.sin(raan) + np.sin(inc/2)*delta_raan*np.cos(raan)   #(G.313)
    d_4 = (np.sin(inc/2) + np.cos(inc/2)*delta_inc/2)*np.cos(raan) - np.sin(inc/2)*delta_raan*np.sin(raan)   #(G.314)

    raan_mean = np.arctan(d_3/d_4)   #(G.315)

    inc_mean = 2*np.arcsin(np.sqrt(d_3**2 + d_4**2))   #(G.316)

    argp_mean = ma_argp_raan_mean - ma_mean - raan_mean   #(G.317)

    inc_mean = inc_mean*180/np.pi
    if raan_mean < 0:
        raan_mean = (raan_mean + np.pi)%np.pi * 180/np.pi
    else:
        raan_mean = raan_mean*180/np.pi
    argp_mean = argp_mean*180/np.pi
    ma_mean = ma_mean*180/np.pi
    
    mean_elements = [a_mean, ecc_mean, inc_mean, raan_mean, argp_mean, ma_mean]

    return mean_elements






