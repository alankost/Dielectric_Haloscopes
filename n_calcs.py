"""
Upated December 31, 2021 to include molecular polarization and local field
Updated January 1, 2022 removing epsilon0 from calculation of molecular
polarizability and including it in calculation of chi
Updated January 1, 2022 putting back epison0 in calculation of molecular
polarizability
Updated January 1, 2022 adding a thermal occupation factor p0 for the lower
transition level
"""
import stacks

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special
from scipy.signal import convolve
mpl.rcParams["axes.formatter.useoffset"] = False

# We need these constants and use MKS units
# They have been calculated using P=1 ATM, T=273K, and Mmolecule=40 mproton
e = 1.602e-19 # electronic charge in Coulomb
m = 9.11e-31 # electronic mass in kg
eps0 = 8.854e-12 # vacuum permitivity in F/m
c = 3.00e8 # speed of light in m/s
gamma_col = 8.4e9 # collision rate in radians per second
delta_omega_Dop = 3.52e9 # Doppler "rate" in radians per second
lambda0 = 10e-6 # vacuum wavelentgh for resonant transision in m
omega0 = 2 * np.pi * c/lambda0
p0 = 1e-6 # thermal occupation factor
N0 = p0*6.022e23*1e6/22.4 # number density in level 0 at T=273 C and P=1 atm
f = 0.057 # oscillator strengh - the "reference" value from paper by Axner
nbg = 1.00 # background refractive index


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
# Voigt broadening line shape function
def gV(omega):
    x = 1.67*(omega - omega0)/delta_omega_Dop
    b = 1.67*gamma_col/delta_omega_Dop
    z = x + 1j*b
    return 5.90/delta_omega_Dop*np.real(np.exp(-z**2)*scipy.special.erfc(-1j*z))/2/np.pi

# real and imaginary parts of the molecular polarizability
def re_gamma(omega):
    return (f*np.pi*e**2/2/m/omega0/eps0)*(omega0-omega)/gamma_col*gcol(omega)
def im_gamma(omega):
    return (f*np.pi*e**2/2/m/omega0/eps0)*gcol(omega)
# total molecular polarizability
def gamma(omega):
    return re_gamma(omega)+1j*im_gamma(omega)
# susceptibility taking into acount local field effects
def chi(omega):
    return N0*gamma(omega)/(1-N0*gamma(omega)/3)

# relative permitivity (dielectric constant)
def epsilon(omega):
    return nbg+chi(omega)

# absorption coefficient and refractive index
def alpha(omega):
    return 2*omega/c*nbg*np.imag(np.sqrt(nbg+chi(omega)))
def n(omega):
    return np.real(np.sqrt(nbg ** 2 + chi(omega)))

def complex_n(omega):
    return np.sqrt(nbg ** 2 + chi(omega))

omega = np.linspace(omega0-50*gamma_col, omega0+50*gamma_col, 1001)
omega_norm = (omega - omega0)/(gamma_col)

omega_wide = np.linspace(0, 2 * omega0, 1001)

A = [0, 1] * 20 + [0]  # control stack size

boost2 = stacks.hw_boost(omega_wide, omega0, 1.5, complex_n, A, n2_real0=1)
plt.plot(omega_wide / omega0, boost2)
plt.gca().set_yscale('log')
plt.ylim(1e-1, 1e2)
plt.xlabel(r'$\omega / \omega_0$')
plt.ylabel(r'boost factor $|B|$')
plt.title(r'$n_1 = 1.5$, $n_2 = n(\omega)$')
plt.show()

boost1 = stacks.hw_boost(omega, omega0, 1.5, complex_n, A, n2_real0=1)
plt.plot(omega_norm, boost1)
plt.gca().set_yscale('log')
plt.ylim(1e-1, 1e2)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel(r'boost factor $|B|$')
plt.title(r'$n_1 = 1.5$, $n_2 = n(\omega)$ (closeup of resonance)')
plt.show()

# test boost factor as a function of refractive index
ntest = np.linspace(0.7, 1.4, 100)
boost3 = [stacks._hw_boost(omega0, omega0, 1.5, n_, 1.5, 1, A, 1, 1) for n_ in ntest]
plt.plot(ntest, boost3)
plt.gca().set_yscale('log')
plt.ylim(1e-1, 1e2)
plt.xlabel(r'$n$')
plt.ylabel(r'boost factor $|B|$')
plt.title(r'boost factor vs. $n$ (with $n$ real)')
plt.show()

# test boost factor as a function of absorption coefficient
ntest = 1 + 1j * np.linspace(-2, 2, 2000)
boost4 = [stacks._hw_boost(omega0, omega0, 1.5, n_, 1.5, 1, A, 1, 1) for n_ in ntest]
plt.plot(np.imag(ntest), boost4)
plt.gca().set_yscale('log')
plt.ylim(1, 200)
plt.xlabel(r'$Im(n)$')
plt.ylabel(r'boost factor $|B|$')
plt.title(r'$|B|$ vs. imaginary part of $n$ (with $Re(n) = 1$)')
plt.show()

plt.plot(omega_norm, gcol(omega))
plt.plot(omega_norm,gV(omega))
plt.plot(omega_norm,gDop(omega))
plt.xlim(-5, 5)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Line Shape Function')
plt.show()

plt.plot(omega_norm, np.real(chi(omega)))
plt.plot(omega_norm, np.imag(chi(omega)))
plt.xlim(-20, 20)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Susceptibility')
plt.show()

plt.plot(omega_norm,alpha(omega))
plt.xlim(-20, 20)
plt.xlabel('Omega - omega0 [gamma_col]')
plt.ylabel('absorption coefficient (m^-1)')
plt.show()

plt.plot(omega_norm, np.exp(-alpha(omega)*0.001))
plt.xlim(-20, 20)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Transmission for 1mm Thickness')
plt.show()

plt.plot(omega_norm,n(omega))
plt.xlim(-20, 20)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('refractive index')
plt.show()

plt.plot(omega_norm,np.real(epsilon(omega)))
plt.plot(omega_norm,np.imag(epsilon(omega)))
plt.xlim(-20, 20)
plt.xlabel('omega - omega0 [gamma_col]')
plt.ylabel('Relative Permitivity')
plt.show()
