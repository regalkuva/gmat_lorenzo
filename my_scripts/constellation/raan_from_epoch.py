from astropy import units as u

from poliastro.earth.util import raan_from_ltan

def raan_from_epoch(ltan, epoch):
    '''This function provides the instantaneous Right Ascension at the Ascending Node (RAAN)  of a 
    Sun Synchronous Orbit (SSO) taking in input the Local Time at the Ascending Node (LTAN) and epoch
    Inputs:
    - ltan [hours]
    - epoch [UTC Gregorian]'''

    raan = raan_from_ltan(epoch, ltan * u.hourangle)

    return raan.wrap_at(360 * u.deg)


# # main
from poliastro.util import Time
epoch  = Time('2025-06-01 00:00', scale = "utc")
ltan   = 22.5
raan = raan_from_epoch(ltan, epoch)
print(raan)