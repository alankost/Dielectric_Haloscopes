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