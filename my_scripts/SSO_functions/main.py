import raan_from_epoch
from poliastro.util import Time

epoch  = Time('2023-03-24 00:00', scale = "utc")
ltan   = 12


raan = raan_from_epoch(ltan, epoch)
