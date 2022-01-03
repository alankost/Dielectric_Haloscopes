"""Using transfer matrices to describe 1D setups with distributed radiation sources."""
import numpy as np


def boost(n, delta, A):
    """General transfer matrix formalism.
    `n`, `delta`, and `A` are arrays, containing values for each region in the stack.

    :param n: refractive index, shape (m + 1,)
    :param delta: (omega n / c) d, shape (m - 1,)
    :param A: (relative) induced electric field, shape (m + 1,)
    :return: boost factor
    """
    # convert to arrays
    n = np.asanyarray(n)
    delta = np.asanyarray(delta)
    A = np.asanyarray(A)

    n_left = n[:-1]
    n_right = n[1:]
    # transfer matrices for single interfaces
    G = ((n_right - n_left) * np.ones((2, 2, 1)) + 2 * n_left * np.eye(2)[..., np.newaxis]) / (2 * n_right)
    # transfer matrices for single regions
    P = np.exp(1j * delta) * np.array([[1, 0], [0, 0]])[..., np.newaxis] \
        + np.exp(-1j * delta) * np.array([[0, 0], [0, 1]])[..., np.newaxis]
    # move last dimension to first
    G = np.moveaxis(G, -1, 0)
    P = np.moveaxis(P, -1, 0)

    Ts = [np.eye(2)]    # "partial" transfer matrices for the setup (in reverse order)
    for Gr, Pr in zip(G[::-1], P[::-1]):
        Ts.append(np.linalg.multi_dot((Ts[-1], Gr, Pr)))
    T = Ts[-1] @ G[0]   # ordinary transfer matrix for entire setup
    S = np.diff(A) / 2  # radiation "sources" at each interface
    M = (Ts * S[::-1, np.newaxis, np.newaxis]).sum(axis=0)  # the "M" matrix

    # boost amplitudes
    BL = -(M[1, 0] + M[1, 1]) / T[1, 1]
    BR = M[0, 0] + M[0, 1] + BL * T[0, 1]
    return np.sqrt(np.abs(BL) ** 2 + np.abs(BR) ** 2)


def _hw_boost(omega, omega0, n1, n2, n1_real0, n2_real0, A, n0=None, nm=None):
    """Does the work for `hw_boost`.

    :param omega: angular frequency
    :param omega0: resonant angular frequency of the half-wave stack
    :param n1: complex refractive index of layer 1, a function of omega
    :param n2: complex refractive index of layer 2, a function of omega
    :param n1_real0: real refractive index of layer 1 on resonance
    :param n2_real0: real refractive index of layer 2 on resonance
    :param A: (relative) induced electric field in each region, shape (m + 1,)
    :param n0: complex refractive index of "left" outer (infinite) region, a function of omega
    :param nm: complex refractive index of "right" outer (infinite) region, a function of omega
    :return: boost factor
    """
    assert len(A) >= 2

    num_layers = len(A) - 2         # number of finite-size layers
    num_pairs = num_layers // 2     # number of pairs of half-wave layers
    n_middle = [n1(omega), n2(omega)] * num_pairs
    delta = [_hw_delta(n1(omega), n1_real0, omega, omega0),
             _hw_delta(n2(omega), n2_real0, omega, omega0)] * num_pairs
    if len(A) % 2 != 0:             # odd number of regions
        n_middle.append(n_middle[0])
        delta.append(delta[0])
    n = [n0(omega)] + n_middle + [nm(omega)]
    return boost(n, delta, A)


def hw_boost(omega, omega0, n1, n2, A, n0=None, nm=None):
    """Boost factor for a half-wave stack with resonant frequency `omega0`.

    :param omega: angular frequency, can be a 1D array
    :param omega0: resonant angular frequency of the half-wave stack
    :param n1: complex refractive index of layer 1, can be a function of omega
    :param n2: complex refractive index of layer 2, can be a function of omega
    :param A: (relative) induced electric field in each region, shape (m + 1,)
    :param n0: complex refractive index of "left" outer (infinite) region, can be a function of omega
    :param nm: complex refractive index of "right" outer (infinite) region, can be a function of omega
    :return: boost factor
    """
    if not n0:
        def n0(omega):
            return 1  # default n0
    elif not callable(n0):
        n0_ = n0
        def n0(omega):
            return n0_
    if not nm:
        def nm(omega):
            return 1  # default nm
    elif not callable(nm):
        nm_ = nm
        def nm(omega):
            return nm_

    # turn n's into functions
    if not callable(n1):
        n1_ = n1
        def n1(omega):
            return n1_
    if not callable(n2):
        n2_ = n2
        def n2(omega):
            return n2_

    # real refractive indices on resonance
    n1_real0 = np.real(n1(omega0))
    n2_real0 = np.real(n2(omega0))

    try:  # iterable omega
        result = np.array([_hw_boost(omega_, omega0, n1, n2, n1_real0, n2_real0, A, n0, nm) for omega_ in omega])
    except TypeError:
        result = _hw_boost(omega, omega0, n1, n2, n1_real0, n2_real0, A, n0, nm)
    return result


def _hw_delta(n, n_real0, omega, omega0):
    """Phase obtained from crossing a single layer of a half-wave stack.

    :param n: complex refractive index for the given frequency
    :param n_real0: real refractive index on resonance
    :param omega: angular frequency
    :param omega0: resonant angular frequency
    :return: phase change
    """
    return np.pi * omega / omega0 * n / n_real0


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    A = [0, 1] * 20 + [0]
    # n = [1, 2, 1, 2, 1]
    # delta = [np.pi] * 3
    # b = boost(n, delta, A)
    omega = np.linspace(0, 2, 10000)
    b = hw_boost(omega, 1, lambda omega: 3, lambda omega: 1, A=A)
    plt.plot(omega, b)
    plt.xlabel(r'$\omega / \omega_0$')
    plt.ylabel(r'boost factor $|B|$')
    plt.gca().set_yscale('log')
    plt.ylim(1e-1, 1e2)
    plt.show()
