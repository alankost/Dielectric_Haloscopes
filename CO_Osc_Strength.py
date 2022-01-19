import numpy as np

e = 1.602e-19 # electronic charge in Coulomb
m = 9.11e-31 # electronic mass in kg
eps0 = 8.854e-12 # vacuum permitivity in F/m
c = 3.00e8 # speed of light in m/s
lambda0 = 10e-6 # vacuum wavelentgh for resonant transision in m
omega0 = 2 * np.pi * c/lambda0
hbar = 1.055e-34 # hbar in mks units
dip_mom = 112.8e-12*e # dipole moment of CO in pm

f =(1/3)*(2*m*omega0)/(hbar*e**2)*dip_mom**2
print('f = ', f)

# Occupation factor for the transition
E0 =  80.7 # Energy of lower level relative
# to the ground level
kBT = 205.7 # kBT at 296K in inverse cm
B = 1.35 # separation of rotation levels in inverse cm
Q = kBT/B # approximate value for partition function
p0=np.exp(-E0/kBT)/Q
print ('Occupation Factor = ',p0)

# relative broading to be compared with 8.9e9/30e14
# = 3e-6 for our "reference" molecule with a 10 micron
# transition
rel_broad = 0.0618/2169.20
print ('Relative Broadening = ','{:.2e}'.format(rel_broad))