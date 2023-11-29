## DISTANCE TRAVELLED BY A SATELLITE
## This code computes the distance travelled by a satellite during a specific number of orbits, considering semimajor axis and eccentricity of each orbit.

import math
import pandas as pd
from scipy.special import ellipe

sat_name = 'RHW'

data = pd.read_csv('RHW_SMA_ECC.txt')

sma_list = data['RHW.Earth.SMA'].tolist()
ecc_list = data['RHW.Earth.ECC'].tolist()

dist = 0
i    = 0 

circ_list = []
for i in range(len(sma_list)):
    circ_of_orbit = 4 * sma_list[i] * ellipe(ecc_list[i]**2)
    circ_list.append(circ_of_orbit)

dist = sum(circ_list)
# Alternative algorithm:

# while i < len(ecc_list):
#    circ = 4*sma_list[i]*ellipe(pow(ecc_list[i],2))   #circumference of an ellipse --> C = 4*sma*ellipe(pow(ecc,2))
#    dist = dist + circ
#    i=i+1

# N.B.: for cycle is more efficient than while cycle

print (f"The distance travelled by {sat_name} was of {dist} km")
print (f"The number of orbits performed by {sat_name} was of {len(sma_list)}")