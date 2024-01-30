import numpy as np
from numba import njit as jit
from poliastro.bodies import Earth
from astropy import units as u
from poliastro.core.elements import rv2coe
from poliastro.constants import rho0_earth, H0_earth
from poliastro.core.propagation import func_twobody
from poliastro.core.perturbations import (
    atmospheric_drag_exponential,
    J2_perturbation,
)
from numba import njit as jit
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value

# parameters of SC body
C_D = 2.2  # dimentionless (any value would do)


A_over_m_low = ((0.01 << u.m**2) / (50 * u.kg)).to_value(u.km**2 / u.kg)  # km^2/kg

B_low = C_D * A_over_m_low

A_over_m_high = ((0.1 << u.m**2) / (50 * u.kg)).to_value(u.km**2 / u.kg)  # km^2/kg

B_high = C_D * A_over_m_high

A_over_m_med = ((0.05 << u.m**2) / (50 * u.kg)).to_value(u.km**2 / u.kg)  # km^2/kg

B_med = C_D * A_over_m_med

# parameters of the atmosphere
rho0 = rho0_earth.to(u.kg / u.km**3).value  # kg/km^3
H0 = H0_earth.to(u.km).value

def a_d(t0, state, k, J2, R, C_D, A_over_m, H0, rho0):
    return J2_perturbation(t0, state, k, J2, R) + atmospheric_drag_exponential(
        t0, state, k, R, C_D, A_over_m, H0, rho0
    )

def perturbations_atm_J2(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = a_d(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m,
        H0=H0,
        rho0=rho0,
        J2=Earth.J2.value,
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def perturbations_atm_low(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = atmospheric_drag_exponential(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_low,
        H0=H0,
        rho0=rho0,
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def perturbations_atm_high(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = atmospheric_drag_exponential(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_high,
        H0=H0,
        rho0=rho0,
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad




## my part
from pyatmos import coesa76

def coesa76_model(state, R, C_D, A_over_m):
    
    H = np.linalg.norm(state[:3])

    v_vec = state[3:]
    v = np.linalg.norm(v_vec)
    
    coesa_geom = coesa76(H - R, 'geometric')
    rho = (coesa_geom.rho)
    rho = (rho*(u.kg/u.m**3)).to_value(u.kg/u.km**3)

    return - 0.5 * rho * C_D * A_over_m * v * v_vec

def pertubations_coesa_low(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa76_model(
        state,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_low
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad


def pertubations_coesa_high(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa76_model(
        state,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_high
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def pertubations_coesa_med(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa76_model(
        state,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_med
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad


def acc_max_vs_min(r_vec,v_vec):

    v = np.linalg.norm(v_vec)
    H = np.linalg.norm(r_vec)

    coesa_geom = coesa76(H - R, 'geometric')
    rho = (coesa_geom.rho)
    rho = (rho*(u.kg/u.m**3)).to_value(u.kg/u.km**3)

    return 3 * (1/2) * C_D * rho * v * v * (A_over_m_high - A_over_m_low) / H


def relative_acc(r_vec,v_vec):
    
    v = np.linalg.norm(v_vec)
    H = np.linalg.norm(r_vec)

    coesa_geom = coesa76(H - R, 'geometric')
    rho = (coesa_geom.rho)
    rho = (rho*(u.kg/u.m**3)).to_value(u.kg/u.km**3)

    return 3 * (1/2) * C_D * rho * v * v * (A_over_m_high - A_over_m_low) / H

def coesa_J2(t0, state, k, J2, R, C_D, A_over_m):
    return J2_perturbation(t0, state, k, J2, R) + coesa76_model(
        state, R, C_D, A_over_m
    )

def perturbations_coesa_J2_high(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa_J2(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_high,
        J2=Earth.J2.value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def perturbations_coesa_J2_low(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa_J2(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_low,
        J2=Earth.J2.value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def perturbations_coesa_J2_med(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = coesa_J2(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m_med,
        J2=Earth.J2.value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad