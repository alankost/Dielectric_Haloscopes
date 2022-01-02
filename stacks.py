"""Using transfer matrices to describe 1D setups with distributed radiation sources."""
import numpy as np


def boost(n, delta, A):
    """
    General transfer matrix formalism.
    n, delta, and A are arrays, containing values for each region in the stack.
    :param n: refractive index, shape (m + 1,)
    :param delta: (omega n / c) d, shape (m - 1,)
    :param A: (relative) induced E field, shape (m + 1,)
    :return: boost factors
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

    Ts = [np.eye(2)]  # "partial" transfer matrices for the setup (in reverse order)
    for Gr, Pr in zip(G[::-1], P[::-1]):
        Ts.append(np.linalg.multi_dot((Ts[-1], Gr, Pr)))
    T = Ts[-1] @ G[0]  # ordinary transfer matrix for entire setup
    S = (A[1:] - A[:-1]) / 2  # radiation "sources" at each interface
    M = (Ts * S[::-1, np.newaxis, np.newaxis]).sum(axis=0)  # the "M" matrix

    # boost amplitudes
    BL = -(M[1, 0] + M[1, 1]) / T[1, 1]
    BR = M[0, 0] + M[0, 1] + BL * T[0, 1]

    return BL, BR


def hw_boost(omega, omega0, n1, n2, A, n0=1, nm=1):
    """Boost factors for a half-wave stack with resonant frequency omega0."""
    assert len(A) >= 2
    n_middle = [n1, n2] * ((len(A) - 2) // 2)
    if len(A) % 2 != 0:  # odd number of regions
        n_middle.append(n1)
    n = [n0] + n_middle + [nm]
    return n


if __name__ == '__main__':
    # n = [1, 2, 1, 2, 1]
    # delta = [np.pi] * 3
    A = [0, 1, 0, 1, 0]
    # b = boost(n, delta, A)
    b = hw_boost(1, 1, 1.5, 1.01, A, n0=3, nm=4)
    print(b)
