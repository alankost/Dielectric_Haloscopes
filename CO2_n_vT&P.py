import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt 
import pandas as pd

# load Hitran data from a file for the C12-O16
# isotope of CO which is 98.6544 % of total
file = open('CO2_data_20JAN2022.csv')
data = genfromtxt(file,delimiter=',')
nu0_HT = data[:,0] # transition center frequencies
S_HT = data[:,1] # integrated line strengths
gamma_HT = data[:,2] # line widths in cm^-1
n_air_HT = data[:,3]
delta_self_HT = data[:,4]
elower_HT = data[:,5]

# load data for partition function for the gas
file2 = open('Partfun_12C16O2.txt')
data2 = genfromtxt(file2)
Q_HT = data2[:,1]

def alphaHT(omega,T,P):
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_HT+delta_self_HT*P
    S=S_HT * ( Q_HT[296]/Q_HT[int(T)]*
            np.exp(-c2*elower_HT/T)/np.exp(-c2*elower_HT/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_HT*gamma_HT*P)
    alpha=np.zeros(omega.size)
    for i in range (nu0.size):
        alpha += (S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1)   
    return alpha

def nimagHT(omega,T,P):
    c_cm=2.9979e10 # speed of light in cm/sec
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_HT+delta_self_HT*P
    S=S_HT * ( Q_HT[296]/Q_HT[int(T)]*
            np.exp(-c2*elower_HT/T)/np.exp(-c2*elower_HT/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_HT*gamma_HT*P)
    nimag=np.zeros(omega.size)
    for i in range (nu0.size):
        nimag += c_cm/omega*(S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1)   
    return nimag
   
def nrealHT(omega,T,P):
    c_cm=2.9979e10 # speed of light in cm/sec
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_HT+delta_self_HT*P
    S=S_HT * ( Q_HT[296]/Q_HT[int(T)]*
            np.exp(-c2*elower_HT/T)/np.exp(-c2*elower_HT/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_HT*gamma_HT*P)
    nreal=np.ones(omega.size)
    for i in range (nu0.size):
        nreal += (c_cm/omega*(nu0[i]-nu)/gamma[i]*
                (S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1))   
    return nreal

# Range of eV & angular frequency over which to calculate alpha & nreal
E_eV_range=np.linspace(0.28,0.30,10000)
e=1.602e-19 # electronic charge
hbar=1.05457e-34 # Plank constant
omega_range=E_eV_range*e/hbar
Tgas = 296 # Temperature of gas in degress Kelvin
Pgas = 1 # Pressure of gas in atm (1 atm = 1.01325)

# molecular density in #/cm^-3 at 296K
N0_HT=6.022e23/22.4/1000*273.15/296 # density of an ideal gas

alpha_HT=alphaHT(omega_range,Tgas,Pgas)
plt.plot(E_eV_range,alpha_HT)
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption Coefficient ($cm^{-1})$')
plt.show()

nimag_HT=nimagHT(omega_range,Tgas,Pgas)
plt.plot(E_eV_range,nimag_HT)
plt.xlabel('Energy (eV)')
plt.ylabel('Imag Part of Refractive Index')
plt.show()

nreal_HT=nrealHT(omega_range,Tgas,Pgas)
plt.plot(E_eV_range,nreal_HT-1)
plt.xlabel('Energy (eV)')
plt.ylabel('Real Part of Refractive Index - 1')
plt.show()
