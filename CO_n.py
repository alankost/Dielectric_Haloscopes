import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#Values from HITRAN for stongest CO absorption line
# at 2169.2 cm^-1 (4.61 micron)

# Density of ideal gas at 298K in cm*-3
Ntot = 6.022e23/22.4*1e-6*298/273

# Transition energy in inverse cm
nu_CO_max = 2169.2

# Integrated line strength
Smax_CO = 4.444e-19 # units are cm^-1/(molecule x cm^-2)

# Self Broadening factor (HWHM of line if the gas
# was pure carbon monoxide)
gamma_CO = 0.068 # units are cm*-1
sigma_max_CO = Smax_CO/np.pi/gamma_CO
alpha_max_CO = Ntot * sigma_max_CO
K_max_CO = alpha_max_CO / (nu_CO_max*2*np.pi)
print(K_max_CO)

nu = np.linspace(-50*gamma_CO,50*gamma_CO,1000)
nu_norm = nu/gamma_CO

# imagninary and real parts of refractive index
def K_CO(nu):
    return K_max_CO*(1/np.pi/gamma_CO)/(1+(nu**2/gamma_CO**2))
def n_CO_minus1(nu):
    return -nu/gamma_CO*K_CO(nu)
    
plt.plot(nu_norm,K_CO(nu))
plt.plot(nu_norm,n_CO_minus1(nu))
plt.xlabel(r'$\nu / \nu_0 - \nu_0$')
plt.ylabel(r'K and n-1')
plt.title(r'real and imaginary parts of refractive index')
plt.show