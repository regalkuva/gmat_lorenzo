## SEMIMAJOR AXIS CALCULATOR FROM TLE

import math

mu = 3.986004418e5     # geocentric gravitational constant [km^3s]
n  = 14.8487149       # mean motion [revolutions per day], only needed input from TLE
                       # n_in = 15.26140049;   n_f = 16.48838571     
sma = pow(mu,1/3) / pow(2*n*math.pi/86400,2/3)     # semimajor axis [km]

print ("Semimajor axis =",sma,"km")