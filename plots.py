"""Plots for Ken's update, Jan 5 2021.

Might take a while to run when the plots are set to a high resolution.
"""
import numpy as np

import n_calcs
import stacks

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use('ggplot')
gg_red = np.array([0.886275, 0.290196, 0.2])
mpl.rcParams['axes.xmargin'] = 0

n_glass = 1       # IoR of glass/solid layer
n_gas0 = 1.001      # IoR of gas layer, off molecular resonance, on half-wave resonance
num_layers = 201    # stack size

# plot |BL|^2 vs. omega (wide shot)

omega_wide = np.linspace(0.95, 1.05, 10001) * n_calcs.omega0
boost_wide = stacks.hw_boost(omega_wide, n_calcs.omega0, n_glass, n_calcs.complex_n, num_layers, n2_real0=n_gas0)
boost_wide2 = stacks.hw_boost(omega_wide, n_calcs.omega0, n_glass, n_gas0, num_layers)
# boost_wide3 = stacks.hw_boost(omega_wide, n_calcs.omega0, n_glass, n_calcs.n, A, n2_real0=n_gas0)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
axes[0].plot(omega_wide / n_calcs.omega0, boost_wide ** 2, label=r'$n_{\mathrm{gas}} = n(\omega)$', lw=1.2)
axes[0].plot(omega_wide / n_calcs.omega0, boost_wide2 ** 2, label=r'$n_{\mathrm{gas}} = 1.001$', lw=1.5, alpha=0.5)
# plt.plot(omega_wide / n_calcs.omega0, boost_wide3)
axes[0].set_yscale('log')
axes[0].set_ylim(1e-1, 3e3)
axes[0].set_xlabel(r'$\omega / \omega_0$')
axes[0].set_ylabel(r'$|\mathrm{B}|^2$')
# plt.title(r'$n_1 = n_glass$, $n_2 = n(\omega)$')
axes[0].legend(loc='upper left')
fig.tight_layout()

# plot |BL|^2 vs. omega (closeup)

omega_closeup_norm = np.linspace(-60, 60, 5001)
omega_closeup = omega_closeup_norm * n_calcs.gamma_col + n_calcs.omega0
boost_closeup = stacks.hw_boost(omega_closeup, n_calcs.omega0, n_glass, n_calcs.complex_n, num_layers, n2_real0=n_gas0)
boost_closeup2 = stacks.hw_boost(omega_closeup, n_calcs.omega0, n_glass, n_gas0, num_layers)
# boost_closeup3 = stacks.hw_boost(omega_closeup, n_calcs.omega0, n_glass, n_calcs.n, A, n2_real0=n_gas0)

axes[1].plot(omega_closeup_norm, boost_closeup ** 2, lw=1.2)
axes[1].plot(omega_closeup_norm, boost_closeup2 ** 2, lw=1.5, alpha=0.5)
# plt.plot(omega_closeup_norm, boost_closeup3)
axes[1].set_xlabel(r'$(\omega - \omega_0) / \gamma_{col}$')
# plt.title(r'$n_1 = n_glass$, $n_2 = n(\omega)$')
plt.savefig('jan5/boost-vs-omega.png')

# plot (magnitude of left) boost factor squared vs. Re(n)

n_real2 = np.linspace(0.8, 1.2, 5001)
n_imags = [0, 0.005, 0.02]
boosts = [np.array([stacks.hw_boost(1, 1, n_glass, n_ + 1j * n_imag_, num_layers, n2_real0=n_gas0) for n_ in n_real2])
          for n_imag_ in n_imags]

plt.figure(figsize=(6.4, 3.6))
colors = [gg_red * scale for scale in (1, 0.6, 0.3)]
linewidths = [1.25, 1.5, 1.75]
plots = [plt.plot(n_real2, boost_ ** 2, label=r'$\mathrm{Im}(n) = ' + str(n_imag_) + '$', c=color, lw=linewidth)
         for n_imag_, boost_, color, linewidth in zip(n_imags, boosts, colors, linewidths)]
plt.gca().set_yscale('log')
plt.ylim(4e-2, 3e3)

plt.xlabel(r'$\mathrm{Re}(n)$')
plt.ylabel(r'$|B|^2$')
# plt.title(r'boost factor vs. $n$ (with $n$ real)')
plt.legend()
plt.tight_layout()
plt.savefig('jan5/boost-vs-n-real.png')

# plot (magnitude of left) boost factor squared vs. Im(n)

n_imag_negative = np.linspace(-0.4, 0, 2001)
n_imag_positive = np.linspace(0.4, 0, 1001)
boost4_negative = np.array([stacks.hw_boost(1, 1, n_glass, 1 + 1j * n_, num_layers, n2_real0=n_gas0)
                            for n_ in n_imag_negative])
boost4_positive = np.array([stacks.hw_boost(1, 1, n_glass, 1 + 1j * n_, num_layers, n2_real0=n_gas0)
                            for n_ in n_imag_positive])

plt.figure(figsize=(6.4, 3.6))
plt.plot(n_imag_positive, boost4_positive ** 2)
plt.plot(n_imag_negative, boost4_negative ** 2, ls='--')
plt.gca().set_yscale('log')
plt.xlabel(r'$\mathrm{Im}(n)$')
plt.ylabel(r'$|B|^2$')
# plt.title(r'boost factor vs. $n$ (with $n$ real)')

plt.axvspan(2 * min(n_imag_negative), 0, alpha=0.3)
plt.text(0.75, 0.75, 'absorption',
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.text(0.25, 0.75, 'stimulated\nemission',
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.ylim(1e-1, 2e6)
plt.xlim(-plt.xlim()[1], plt.xlim()[1])
plt.tight_layout()
plt.savefig('jan5/boost-vs-n-imag.png')

# 2D color plot of |B|^2 vs. n_real and n_imag

n_real = np.linspace(0.8, 1.2, 500)
n_imag = np.linspace(0, 0.1, 2000)
boost_grid = np.array([[stacks.hw_boost(1, 1, n_glass, n_real_ + 1j * n_imag_, num_layers, n2_real0=1)
                        for n_real_ in n_real] for n_imag_ in n_imag])

plt.figure()
plt.imshow(boost_grid ** 2,
           extent=(n_real[0], n_real[-1], n_imag[0], n_imag[-1]), origin='lower', norm=LogNorm(vmin=1e-2))
cbar = plt.colorbar(location='bottom')
cbar.set_label(r'$|\mathrm{B}|^2$')
plt.xlabel(r'$\mathrm{Re}(n)$')
plt.ylabel(r'$\mathrm{Im}(n)$')
# plt.title(r'$|\mathrm{B}|^2$ vs. $\mathrm{Re}(n)$, $\mathrm{Im}(n)$')
plt.tight_layout()
plt.savefig('jan5/2D-boost.png')

# plot complex refractive index

omega_closeup_norm = np.linspace(-8, 8, 5001)
omega_closeup = omega_closeup_norm * n_calcs.gamma_col + n_calcs.omega0
complex_n = n_calcs.complex_n(omega_closeup)

plt.figure(figsize=(6.4, 3.6))
plt.plot(omega_closeup_norm, np.real(complex_n) - 1, label=r'$\mathrm{Re}(n) - 1$')
plt.plot(omega_closeup_norm, np.imag(complex_n), label=r'$\mathrm{Im}(n)$')
plt.xlabel(r'$(\omega - \omega_0) / \gamma_{col}$')
# plt.ylim(plt.ylim()[0] - 0.1, plt.ylim()[1] + 0.1)
plt.legend()
plt.tight_layout()
plt.savefig('jan5/n-vs-omega.png')

plt.show()
