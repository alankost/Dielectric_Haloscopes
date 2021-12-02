import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.signal import convolve

# We need these constants and use MKS units
# They have been calculated using P=1 ATM, T=273K, and Mmolecule=40 mproton
e = 1.602e-19 # electronic charge in Coulomb
m = 9.11e-31 # electronic mass in kg
eps0 = 8.854e-12 # vacuum permitivity in F/m
c = 3.00e8 # speed of light in m/s
gamma_col = 8.4e9 # collision rate in radians per second
delta_omega_Dop = 3.52e9 # Doppler "rate" in radians per second
lambda0 = 1e-6 # vacuum wavelentgh for resonant transision in m
omega0 = c/lambda0
N0 = 6.023e23*1e3/22.4 # number density in level 0 at T=273 C and P=1 atm
f = 0.057 # oscillator strengh - the "reference" value from paper by Axner
nbg = 1.001 # background refractive index


# Lorentzian broadening line shape function
def gcol(omega):
    return (1/np.pi/gamma_col)*(1/(1+(omega-omega0)**2/(gamma_col)**2))
def gDop(omega):
    return ((1/delta_omega_Dop)*(np.sqrt(4*np.log(2)/np.pi))*
        np.exp(-4*(omega-omega0)**2*np.log(2)/delta_omega_Dop**2))
def wplot(omega):
    x = 1.67*(omega - omega0)/delta_omega_Dop
    b = 1.67*gamma_col/delta_omega_Dop
    z = x + 1j*b
    return np.real(np.exp(-z**2)*scipy.special.erfc(-1j*z))
# Voigt broadenign line shape function
def gV(omega):
    x = 1.67*(omega - omega0)/delta_omega_Dop
    b = 1.67*gamma_col/delta_omega_Dop
    z = x + 1j*b
    return 5.90/delta_omega_Dop*np.real(np.exp(-z**2)*scipy.special.erfc(-1j*z))/2/np.pi

# real and imaginary parts of the susceptibility
def re_chi(omega):
    return (f*N0*np.pi*e**2/2/m/eps0/omega0)*(omega0-omega)/gamma_col*gcol(omega)
def im_chi(omgea):
    return (f*N0*np.pi*e**2/2/m/eps0/omega0)*gcol(omega)
# total susceptibility
def chi(omega):
    return re_chi(omega)+1j*im_chi(omega)

# absorption coefficient and refractive index
def alpha(omega):
    return np.imag((2*omega/c)*nbg*np.sqrt(1+(re_chi(omega)+im_chi(omega)*1j/nbg**2)))
def n(omgea):
    return np.real(nbg*np.sqrt(1+(re_chi(omega)+im_chi(omega)*1j/nbg**2)))
# absorption coefficient and refractive index from Clausius-Mossoti relation
def alpha_local(omega):
    return 2*omega/c*np.imag(np.sqrt((1+2/3*((nbg-1)**2*chi(omega)))/(1-1/3*((nbg-1)**2+chi(omega)))))
def n_local(omega):
    return np.real(np.sqrt((1+2/3*((nbg-1)**2*chi(omega)))/(1-1/3*((nbg-1)**2+chi(omega)))))
omega = np.linspace(omega0-5*gamma_col, omega0+5*gamma_col, 101)
omega_norm = (omega - omega0)/(gamma_col)

"""
plt.plot(omega_norm, gcol(omega))
plt.plot(omega_norm, convolve(gcol(omega),gDop(omega),'same')*gamma_col/10)
plt.plot(omega_norm,gV(omega))
plt.plot(omega_norm,gDop(omega))
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Convolution')
plt.show()

plt.plot(omega_norm, re_chi(omega))
plt.plot(omega_norm, im_chi(omega))
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Susecptibility')
plt.show()
"""

plt.plot(omega_norm,alpha(omega))
# plt.plot(omega_norm,alpha_local(omega))
plt.xlabel('Omega - omega0 [gamma_col]')
plt.ylabel('absorption coefficient (m^-1)')
plt.show()

plt.plot(omega_norm,n(omega))
# plt.plot(omega_norm,n_local(omega))
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('refractive index')
plt.show()

plt.plot(omega_norm, np.exp(-alpha(omega)*lambda0/n(omega)/2))
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Transmission for Half_wavelength Layer')
plt.show()

plt.plot (omega_norm, lambda0/n(omega)/2*1e6)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Half_wave Thickness (micron)')
plt.show()