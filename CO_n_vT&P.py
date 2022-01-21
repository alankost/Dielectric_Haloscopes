import numpy as np
from numpy import loadtxt, genfromtxt
import matplotlib.pyplot as plt

# load data from a file that contains three columns - transition energy in cm-1,
# integrated line stregnth in cm^-1 per molecule per cm^-2,
# and line width in cm^-1
# data is for the C12-O16 isotope of CO which is 98.6544 % of total
# data is also for T=296K 
file = open('CO_data_20JAN2022.txt')
data = genfromtxt(file)
nu0_CO = data[:,0] # transition center frequencies
S_CO = data[:,1] # integrated line strengths
gamma_CO = data[:,2] # line widths in cm^-1
n_air_CO = data[:,3]
delta_self_CO = data[:,4]
elower_CO = data[:,5]

# load data for partition function for CO
file2 = open('Partfun_12C16O.txt')
data2 = genfromtxt(file2)
Q_CO = data2[:,1]

def alphaCO(omega,T,P):
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_CO+delta_self_CO*P
    S=S_CO * ( Q_CO[296]/Q_CO[int(T)]*
            np.exp(-c2*elower_CO/T)/np.exp(-c2*elower_CO/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_CO*gamma_CO*P)
    alpha=np.zeros(omega.size)
    for i in range (nu0.size):
        alpha += (S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1)   
    return alpha

def nimagCO(omega,T,P):
    c_cm=2.9979e10 # speed of light in cm/sec
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_CO+delta_self_CO*P
    S=S_CO * ( Q_CO[296]/Q_CO[int(T)]*
            np.exp(-c2*elower_CO/T)/np.exp(-c2*elower_CO/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_CO*gamma_CO*P)
    nimag=np.zeros(omega.size)
    for i in range (nu0.size):
        nimag += c_cm/omega*(S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1)   
    return nimag
   
def nrealCO(omega,T,P):
    c_cm=2.9979e10 # speed of light in cm/sec
    N0=6.022e23/22.4/1000*273.15/T
    c2=1.4387769
    nu = omega/2/np.pi/2.998e10
    nu0 = nu0_CO+delta_self_CO*P
    S=S_CO * ( Q_CO[296]/Q_CO[int(T)]*
            np.exp(-c2*elower_CO/T)/np.exp(-c2*elower_CO/296)*
            (1-np.exp(-c2*nu0/T))/(1-np.exp(-c2*nu0/296)) )
    gamma = ((296/T)**n_air_CO*gamma_CO*P)
    nreal=np.ones(omega.size)
    for i in range (nu0.size):
        nreal += (c_cm/omega*(nu0[i]-nu)/gamma[i]*
                (S[i]*N0/np.pi/gamma[i]*
                   (1+(nu-nu0[i])**2/gamma[i]**2)**-1))   
    return nreal

# Range of eV & angular frequency over which to calculate alpha & nreal
E_eV_range=np.linspace(0.260,0.27,10000)
e=1.602e-19 # electronic charge
hbar=1.05457e-34 # Plank constant
omega_range=E_eV_range*e/hbar
T_CO = 100 # Temperature of CO in degress Kelvin
P_CO = 5 # Pressure of CO in atm (1 atm = 1.01325)

# molecular density in #/cm^-3 at 296K
N0_CO=6.022e23/22.4/1000*273.15/296 # density of an ideal gas

alpha_CO=alphaCO(omega_range,T_CO,P_CO)
plt.plot(E_eV_range,alpha_CO)
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption Coefficient ($cm^{-1})$')
plt.show()

nimag_CO=nimagCO(omega_range,T_CO,P_CO)
plt.plot(E_eV_range,nimag_CO)
plt.xlabel('Energy (eV)')
plt.ylabel('Imag Part of Refractive Index')
plt.show()

nreal_CO=nrealCO(omega_range,T_CO,P_CO)
plt.plot(E_eV_range,nreal_CO-1)
plt.xlabel('Energy (eV)')
plt.ylabel('Real Part of Refractive Index - 1')
plt.show()