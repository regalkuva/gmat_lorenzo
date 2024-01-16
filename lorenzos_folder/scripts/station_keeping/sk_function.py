from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody.thrust import change_a_inc
from poliastro.core.propagation import func_twobody 
from poliastro.core.perturbations import J2_perturbation

from perturbations_sk import coesa76_model
from sso_inc_sk import inc_from_alt



a_d_thrust, deltaV, t_f = change_a_inc(k, a_down, a_up, inc_down, inc_up, acc)

