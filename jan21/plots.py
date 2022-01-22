"""Plots for Ken's update, Jan 21 2021.

Essentially a copy of `plots` for the jan 5 update, but updated to use CO.

Might take a while to run when the plots are set to a high resolution.
"""
import sys
sys.path.append('..')

import CO_n_v_TP
import stacks

import functools
import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use('ggplot')
gg_red = np.array([0.886275, 0.290196, 0.2])
mpl.rcParams.update({'axes.xmargin': 0, 'figure.dpi': 200})

n_glass = 1.5       # IoR of glass/solid layer
n_gas0 = 1          # IoR of gas layer, off molecular resonance, on half-wave resonance
num_layers = 201    # stack size

CO_n_complex = functools.partial(CO_n_v_TP.ncomplexHT, T=100, P=5)
eV_to_omega = 1.519e15
omega0 = 0.266 * eV_to_omega

start_time = time.perf_counter()

# plot |BL|^2 vs. omega (wide shot)

omega_wide = np.linspace(0.5, 1.5, 10001) * omega0
boost_wide = stacks.hw_boost(omega_wide, omega0, n_glass, CO_n_complex, num_layers, n2_real0=n_gas0)
boost_wide2 = stacks.hw_boost(omega_wide, omega0, n_glass, n_gas0, num_layers)
# boost_wide3 = stacks.hw_boost(omega_wide, n_calcs.omega0, n_glass, n_calcs.n, A, n2_real0=n_gas0)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
axes[0].plot(omega_wide / omega0, boost_wide ** 2, label=r'$n_{\mathrm{gas}} = n(\omega)$', lw=1.2)
axes[0].plot(omega_wide / omega0, boost_wide2 ** 2, label=r'$n_{\mathrm{gas}} = 1.001$', lw=1.5, alpha=0.5)
# plt.plot(omega_wide / n_calcs.omega0, boost_wide3)
axes[0].set_yscale('log')
axes[0].set_ylim(1e-1, 3e3)
axes[0].set_xlabel(r'$\omega / \omega_0$')
axes[0].set_ylabel(r'$|\mathrm{B}|^2$')
# plt.title(r'$n_1 = n_glass$, $n_2 = n(\omega)$')
axes[0].legend(loc='upper left')
fig.tight_layout()

# # plot |BL|^2 vs. omega (closeup)
#
# omega_closeup_norm = np.linspace(-60, 60, 5001)
# omega_closeup = omega_closeup_norm * n_calcs.gamma_col + n_calcs.omega0
# boost_closeup = stacks.hw_boost(omega_closeup, omega0, n_glass, CO_n_complex, num_layers, n2_real0=n_gas0)
# boost_closeup2 = stacks.hw_boost(omega_closeup, omega0, n_glass, n_gas0, num_layers)
# # boost_closeup3 = stacks.hw_boost(omega_closeup, n_calcs.omega0, n_glass, n_calcs.n, A, n2_real0=n_gas0)
#
# axes[1].plot(omega_closeup_norm, boost_closeup ** 2, lw=1.2)
# axes[1].plot(omega_closeup_norm, boost_closeup2 ** 2, lw=1.5, alpha=0.5)
# # plt.plot(omega_closeup_norm, boost_closeup3)
# axes[1].set_xlabel(r'$(\omega - \omega_0) / \gamma_{col}$')
# # plt.title(r'$n_1 = n_glass$, $n_2 = n(\omega)$')

plt.savefig('boost-vs-omega.png')

# # plot (magnitude of left) boost factor squared vs. Re(n)
#
# n_real2 = np.linspace(0.8, 1.2, 5001)
# n_imags = np.array([0, 0.005, 0.02])[:, np.newaxis]
# boosts = stacks.hw_boost(1, 1, n_glass, n_real2 + 1j * n_imags, num_layers, n2_real0=n_gas0)
#
# plt.figure(figsize=(6.4, 3.6))
# colors = [gg_red * scale for scale in (1, 0.6, 0.3)]
# linewidths = [1.25, 1.5, 1.75]
# plots = [plt.plot(n_real2, boost_ ** 2, label=r'$\mathrm{Im}(n) = ' + str(n_imag_) + '$', c=color, lw=linewidth)
#          for n_imag_, boost_, color, linewidth in zip(n_imags, boosts, colors, linewidths)]
# plt.gca().set_yscale('log')
# plt.ylim(4e-2, 3e3)
#
# plt.xlabel(r'$\mathrm{Re}(n)$')
# plt.ylabel(r'$|B|^2$')
# # plt.title(r'boost factor vs. $n$ (with $n$ real)')
# plt.legend()
# plt.tight_layout()
# plt.savefig('boost-vs-n-real.png')
#
# # plot (magnitude of left) boost factor squared vs. Im(n)
#
# n_imag_negative = np.linspace(-0.4, 0, 2001)
# n_imag_positive = np.linspace(0.4, 0, 1001)
# boost4_negative = stacks.hw_boost(1, 1, n_glass, 1 + 1j * n_imag_negative, num_layers, n2_real0=n_gas0)
# boost4_positive = stacks.hw_boost(1, 1, n_glass, 1 + 1j * n_imag_positive, num_layers, n2_real0=n_gas0)
#
# plt.figure(figsize=(6.4, 3.6))
# plt.plot(n_imag_positive, boost4_positive ** 2)
# plt.plot(n_imag_negative, boost4_negative ** 2, ls='--')
# plt.gca().set_yscale('log')
# plt.xlabel(r'$\mathrm{Im}(n)$')
# plt.ylabel(r'$|B|^2$')
# # plt.title(r'boost factor vs. $n$ (with $n$ real)')
#
# plt.axvspan(2 * min(n_imag_negative), 0, alpha=0.3)
# plt.text(0.75, 0.75, 'absorption',
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.text(0.25, 0.75, 'stimulated\nemission',
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
#
# plt.ylim(1e-1, 2e6)
# plt.xlim(-plt.xlim()[1], plt.xlim()[1])
# plt.tight_layout()
# plt.savefig('boost-vs-n-imag.png')
#
# # 2D color plot of |B|^2 vs. n_real and n_imag, wide shot
#
# # things shared by the closeup shot
# resolution = 1000
# alpha_box = 0.45
# alpha_arrow1 = 0.6
# alpha_arrow2 = 0.5
# n_line_alpha = 0.5
# n_linewidth = 1.25
# omega_for_n = np.linspace(-60, 60, 10000) * n_calcs.gamma_col + n_calcs.omega0
# n_of_omega = n_calcs.complex_n(omega_for_n)
#
# n_real1 = np.linspace(0.8, 1.2, resolution)
# n_imag1 = np.linspace(0, 0.4, resolution)
# boost_grid = np.vstack([stacks.hw_boost(1, 1, n_glass, n_real1 + 1j * n_imag_, num_layers, n2_real0=n_gas0)
#                         for n_imag_ in np.array_split(n_imag1[:, np.newaxis], 100)])
#
# n_of_omega0 = n_calcs.complex_n(n_calcs.omega0)
# n_of_omega0_xy = np.array([np.real(n_of_omega0), np.imag(n_of_omega0)])
# n_resonance = np.array([1, max(np.imag(n_of_omega))])
#
# fig2, axes2 = plt.subplots()
# imshow = axes2.imshow(boost_grid ** 2, extent=(n_real1[0], n_real1[-1], n_imag1[0], n_imag1[-1]), origin='lower',
#                       norm=LogNorm(vmin=1e-2))
# axes2.plot(np.real(n_of_omega), np.imag(n_of_omega), c='red', alpha=n_line_alpha, label=r'$n(\omega)$', lw=n_linewidth)
# cbar = fig2.colorbar(imshow, location='right')
# cbar.set_label(r'$|\mathrm{B}|^2$')
# axes2.scatter(*n_of_omega0_xy, c='orange', zorder=2)
# axes2.scatter(*n_resonance, c='orange', zorder=2)
# axes2.annotate(r'$\omega_0$', n_of_omega0_xy, xytext=n_of_omega0_xy + (0, -0.04), ha='center', va='center',
#                arrowprops={'facecolor': 'gray', 'alpha': alpha_arrow1, 'width': 3},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes2.annotate(r'$\omega_\mathrm{res}$', n_resonance, xytext=n_resonance + (0.02, -0.04),
#                ha='center', va='center',
#                arrowprops={'facecolor': 'gray', 'alpha': alpha_arrow1, 'width': 3},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes2.annotate(r'increasing $\omega$', (0.84, 0.22), xytext=(0.91, 0.29), ha='center', va='center',
#                arrowprops={'facecolor': 'red', 'alpha': alpha_arrow2, 'width': 3, 'connectionstyle': 'arc3,rad=0.3'},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes2.annotate(r'decreasing $\omega$', (2 - 0.84, 0.22), xytext=(2 - 0.91, 0.29), ha='center', va='center',
#                arrowprops={'facecolor': 'red', 'alpha': alpha_arrow2, 'width': 3, 'connectionstyle': 'arc3,rad=-0.3'},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
#
# axes2.set_xlabel(r'$\mathrm{Re}(n)$')
# axes2.set_ylabel(r'$\mathrm{Im}(n)$')
# # plt.title(r'$|\mathrm{B}|^2$ vs. $\mathrm{Re}(n)$, $\mathrm{Im}(n)$')
# axes2.legend()
# fig2.tight_layout()
# fig2.savefig('2D-boost-wide.png')
#
# # closeup shot
#
# n_real2 = np.linspace(0.9, 1.1, resolution)
# n_imag2 = np.linspace(0, 0.02, resolution)
# boost_grid2 = np.vstack([stacks.hw_boost(1, 1, n_glass, n_real2 + 1j * n_imag_, num_layers, n2_real0=n_gas0)
#                          for n_imag_ in np.array_split(n_imag2[:, np.newaxis], 100)])
# # omega_for_n2 = np.linspace(-1000, 1000, 10000) * n_calcs.gamma_col + n_calcs.omega0
# n_of_omega2 = n_of_omega
# n_omega_min = np.array([np.real(n_of_omega2[0]), np.imag(n_of_omega2[0])])
# n_omega_max = np.array([np.real(n_of_omega2[-1]), np.imag(n_of_omega2[-1])])
#
# fig3, axes3 = plt.subplots()
# axes3.imshow(boost_grid2 ** 2,
#              extent=(n_real2[0], n_real2[-1], n_imag2[0], n_imag2[-1]), origin='lower', norm=LogNorm(vmin=1e-2))
# axes3.plot(np.real(n_of_omega2), np.imag(n_of_omega2), c='red', alpha=n_line_alpha, label=r'$n(\omega)$',
#            lw=n_linewidth)
# axes3.scatter(np.real(n_of_omega2[[0, -1]]), np.imag(n_of_omega2[[0, -1]]), c='black', zorder=2)
# axes3.annotate(r'$\omega = \omega_0 - 60 \gamma_\mathrm{col}$', n_omega_min, xytext=n_omega_min + (0.0035 * 5/3.2, 0.005),
#                ha='center', va='center',
#                arrowprops={'facecolor': 'gray', 'alpha': alpha_arrow1, 'width': 3},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes3.annotate(r'$\omega = \omega_0 + 60 \gamma_\mathrm{col}$', n_omega_max, xytext=n_omega_max + (0.0035, 0.0032),
#                ha='center', va='center',
#                arrowprops={'facecolor': 'gray', 'alpha': alpha_arrow1, 'width': 3},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes3.annotate(r'increasing $\omega$', (0.955, 0.0078), xytext=(0.945, 0.011), ha='center', va='center',
#                arrowprops={'facecolor': 'red', 'alpha': alpha_arrow2, 'width': 3, 'connectionstyle': 'arc3,rad=0.1'},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
# axes3.annotate(r'decreasing $\omega$', (2 - 0.955, 0.0078), xytext=(2 - 0.945, 0.011), ha='center', va='center',
#                arrowprops={'facecolor': 'red', 'alpha': alpha_arrow2, 'width': 3, 'connectionstyle': 'arc3,rad=-0.1'},
#                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': alpha_box})
#
# axes3.set_xlabel(r'$\mathrm{Re}(n)$')
# axes3.set_ylabel(r'$\mathrm{Im}(n)$')
# axes3.set_xlim(n_real2[[0, -1]])
# axes3.set_ylim(n_imag2[[0, -1]])
# axes3.set_aspect(10)
# axes3.legend(loc='upper right')
# fig3.tight_layout()
# fig3.savefig('2D-boost-closeup.png')
#
# # plot complex refractive index
#
# omega_closeup_norm = np.linspace(-8, 8, 5001)
# omega_closeup = omega_closeup_norm * n_calcs.gamma_col + n_calcs.omega0
# complex_n = n_calcs.complex_n(omega_closeup)
#
# plt.figure(figsize=(6.4, 3.6))
# plt.plot(omega_closeup_norm, np.real(complex_n) - 1, label=r'$\mathrm{Re}(n) - 1$')
# plt.plot(omega_closeup_norm, np.imag(complex_n), label=r'$\mathrm{Im}(n)$')
# plt.xlabel(r'$(\omega - \omega_0) / \gamma_{col}$')
# # plt.ylim(plt.ylim()[0] - 0.1, plt.ylim()[1] + 0.1)
# plt.legend()
# plt.tight_layout()
# plt.savefig('n-vs-omega.png')

elapsed_time = time.perf_counter() - start_time
print('Done in: {} seconds!'.format(elapsed_time))

plt.show()
