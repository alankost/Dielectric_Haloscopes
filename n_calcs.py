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
lambda0 = 1e-6 # vacuum wavelentgh for resonant transision in m
omega0 = c/lambda0
N0 = 6.022e23*1e6/22.4 # number density in level 0 at T=273 C and P=1 atm
f = 0.57 # oscillator strengh - the "reference" value from paper by Axner
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
# Voigt broadenign line shape function
def gV(omega):
    x = 1.67*(omega - omega0)/delta_omega_Dop
    b = 1.67*gamma_col/delta_omega_Dop
    z = x + 1j*b
    return 5.90/delta_omega_Dop*np.real(np.exp(-z**2)*scipy.special.erfc(-1j*z))/2/np.pi

# real and imaginary parts of the molecular polarizability
def re_gamma(omega):
    return (f*np.pi*e**2/2/m/omega0)*(omega0-omega)/gamma_col*gcol(omega)+(nbg**2-1)
def im_gamma(omgea):
    return (f*np.pi*e**2/2/m/omega0)*gcol(omega)
# total molecular polarizability
def gamma(omega):
    return re_gamma(omega)+1j*im_gamma(omega)
# susceptibility taking into acount local field effects
def chi(omega):
    return N0*gamma(omega)/(1-N0*gamma(omega)/3)

# absorption coefficient and refractive index
def alpha(omega):
    return 2*omega/c*nbg*np.imag(np.sqrt(nbg+chi(omega)))
def n(omgea):
    return np.real(np.sqrt(nbg+chi(omega)))

omega = np.linspace(omega0-50*gamma_col, omega0+50*gamma_col, 1001)
omega_norm = (omega - omega0)/(gamma_col)

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
plt.ylabel('Susecptibility')
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