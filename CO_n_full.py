import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

# load data from a file that contains three columns - transition energy in cm-1,
# integrated line stregnth in cm^-1 per molecule per cm^-2,
# and line width in cm^-1
# data is for the C12-O16 isotope of CO which is 98.6544 % of total
# data is also for T=296K 
file = open('CO_data.csv')
data = loadtxt(file,delimiter = ",")
lmbda=1e-2 # wavlength for 1 inverse cm
c=2.998e8 # speed of light in m/sec
hbar=1.05457e-34 # Plank constant
omega0_CO = data[:,0]*c/lmbda*2*np.pi # transition center frequencies
S_CO = data[:,1] # integrated line strengths
delta_nu_CO = data[:,2] # line widths in cm^-1
delta_omega_CO = delta_nu_CO*c/lmbda*2*np.pi # line widths in rad/sec

def alphaCO(omega,N0):
    alpha=np.zeros(omega.size)
    for i in range (omega0_CO.size):
        alpha += (S_CO[i]*N0/np.pi/delta_nu_CO[i]*
                   (1+(omega-omega0_CO[i])**2/delta_omega_CO[i]**2)**-1)   
    return alpha

def nimagCO(omega,N0):
    nimag=np.zeros(omega.size)
    for i in range (omega0_CO.size):
        nimag += c/omega * (S_CO[i]*N0/np.pi/delta_nu_CO[i]*
                   (1+(omega-omega0_CO[i])**2/delta_omega_CO[i]**2)**-1)   
    return nimag
   
def nrealCO(omega,N0):
    nreal=np.ones(omega.size)
    for i in range (omega0_CO.size):
        nreal += (c/omega*(omega0_CO[i]-omega)/(delta_omega_CO[i])*
                (S_CO[i]*N0/np.pi/delta_nu_CO[i]*
                   (1+(omega-omega0_CO[i])**2/delta_omega_CO[i]**2)**-1))   
    return nreal

# Range of eV & angular frequency over which to calculate alpha & nreal
E_eV_range=np.linspace(0.260,0.262,10000)
e=1.602e-19 # electronic charge
hbar=1.05457e-34 # Plank constant
omega_range=E_eV_range*e/hbar

# molecular density in #/cm^-3 at 296K
N0_CO=6.022e23/22.4/1000*273.15/296 # density of an ideal gas

alpha_CO=alphaCO(omega_range,N0_CO)
plt.plot(E_eV_range,alpha_CO)
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption Coefficient ($cm^{-1})$')
plt.show()

nimag_CO=nimagCO(omega_range,N0_CO)
plt.plot(E_eV_range,nimag_CO)
plt.xlabel('Energy (eV)')
plt.ylabel('Imag Part of Refractive Index')
plt.show()

nreal_CO=nrealCO(omega_range,N0_CO)
plt.plot(E_eV_range,nreal_CO-1)
plt.xlabel('Energy (eV)')
plt.ylabel('Real Part of Refractive Index - 1')
plt.show()