"""Using transfer matrices to describe 1D setups with distributed radiation sources.

TODO: Double-check that Millar's formalism works for complex refractive indices.
"""
import numpy as np


def boost(n, delta, A):
    """General transfer matrix formalism.
    `n`, `delta`, and `A` are arrays, containing values for each region in the stack.
    Uses the convention that a positive imaginary part of the refractive index corresponds to absorption (same
    convention that Millar uses).

    In the shapes below, "x" stands for a common arbitrary shape, or a similar broadcastable shape.

    :param n: refractive index, shape (x, m + 1)
    :param delta: (omega n / c) d, shape (x, m - 1)
    :param A: (relative) induced electric field, shape (x, m + 1)
    :return: magnitude of left boost factor
    """
    # convert to arrays
    n = np.asanyarray(n)
    delta = np.asanyarray(delta)
    A = np.asanyarray(A)

    # transfer matrices for single interfaces
    n_left = n[..., :-1]
    n_right = n[..., 1:]
    n_ratio = (n_left / n_right)[..., np.newaxis, np.newaxis]   # shape (x, m, 1, 1)
    G = (1 + n_ratio * np.array([[1, -1], [-1, 1]])) / 2        # shape (x, m, 2, 2)
    # transfer matrices for single regions
    delta_matrix = delta[..., np.newaxis, np.newaxis]           # shape (x, m - 1, 1, 1)
    P = np.exp(1j * delta_matrix) * np.array([[1, 0], [0, 0]]) + np.exp(-1j * delta_matrix) * np.array([[0, 0], [0, 1]])

    Tmm_x_shape = np.broadcast(n[..., 0], delta[..., 0]).shape
    Tmm = np.broadcast_to(np.eye(2), Tmm_x_shape + (2, 2))      # shape (x, 2, 2)
    Ts = [Tmm]  # "partial" transfer matrices for the setup, T_r^m, in reverse order
    for Gr, Pr in zip(np.moveaxis(G, -3, 0)[::-1], np.moveaxis(P, -3, 0)[::-1]):
        Ts.append(Ts[-1] @ Gr @ Pr)
    Ts = np.stack(Ts, axis=-3)                  # shape (x, m, 2, 2)
    T = Ts[..., -1, :, :] @ G[..., 0, :, :]     # ordinary transfer matrix for entire setup, shape (x, 2, 2)
    # radiation "sources" at each interface, also in reverse order, shape (x, m, 1, 1)
    S = np.diff(A, axis=-1)[..., ::-1, np.newaxis, np.newaxis] / 2
    M = (Ts * S).sum(axis=-3)                   # shape (x, 2, 2)

    # boost amplitudes
    BL = -(M[..., 1, 0] + M[..., 1, 1]) / T[..., 1, 1]
    # BR = M[0, 0] + M[0, 1] + BL * T[0, 1]
    return np.abs(BL)


def hw_boost(omega, omega0, n1, n2, num_layers, n1_real0=None, n2_real0=None, n0=1, nm=1):
    """Boost factor for a half-wave stack with resonant frequency `omega0`. The induced electric field `A` is calculated
    automatically by the `_A` function.

    `omega`, `omega0`, `n1`, `n2`, `n1_real0`, `n2_real0`, `n0`, and `nm` can all be broadcastable to a common shape
    (x,), which will be the shape of the output. Alternatively, `n1`, `n2`, `n0`, and/or `nm` can be functions of
    `omega`. Finally, each n_real0 defaults to Re(n(omega)) or Re(n).

    A positive imaginary part of a refractive index corresponds to absorption.

    :param omega: angular frequency
    :param omega0: resonant angular frequency of the half-wave stack
    :param n1: complex refractive index of layer 1 (for the given omega)
    :param n2: complex refractive index of layer 2 (for the given omega)
    :param num_layers: number of (finite) layers in the stack
    :param n1_real0: real refractive index of layer 1 on resonance (sets width of layer 1)
    :param n2_real0: real refractive index of layer 2 on resonance (sets width of layer 2)
    :param n0: complex refractive index of "left" outer (infinite) region (for the given omega)
    :param nm: complex refractive index of "right" outer (infinite) region (for the given omega)
    :return: boost factor
    """
    # default `n1_real0` and `n2_real0`
    if not n1_real0:
        try:
            n1_real0 = n1(omega0).real
        except TypeError:
            n1_real0 = n1.real
    if not n2_real0:
        try:
            n2_real0 = n2(omega0).real
        except TypeError:
            n2_real0 = np.real(n2)

    # handle callable `n1` and `n2`
    try:  # broadcastable
        delta1 = _hw_delta(n1, n1_real0, omega, omega0)
    except TypeError:  # `n1` should be a function
        n1 = n1(omega)
        delta1 = _hw_delta(n1, n1_real0, omega, omega0)
    try:  # broadcastable
        delta2 = _hw_delta(n2, n2_real0, omega, omega0)
    except TypeError:  # `n2` should be a function
        n2 = n2(omega)
        delta2 = _hw_delta(n2, n2_real0, omega, omega0)
    delta = _hw_alternate(delta1, delta2, num_layers)  # shape (x, `num_layers`)

    # check for callable `n0` and `nm`
    try:
        n0 = n0(omega)
    except TypeError:
        pass
    try:
        nm = nm(omega)
    except TypeError:
        pass

    n = _hw_alternate(n1, n2, num_layers + 2, n0, nm)  # shape (x, `num_layers` + 2)
    A = _A(n)                                          # shape (x, `num_layers` + 2)
    return boost(n, delta, A)


def _hw_delta(n, n_real0, omega, omega0):
    """Phase obtained from crossing a single layer of a half-wave stack.

    :param n: complex refractive index for the given frequency
    :param n_real0: real refractive index on resonance
    :param omega: angular frequency
    :param omega0: resonant angular frequency
    :return: phase change
    """
    return np.pi * omega / omega0 * n / n_real0


def _A(n):
    """Electric field amplitude induced by an oscillating driving interaction in medium with spatially uniform
    refractive index `n`, relative to the electric field amplitude that would create an equivalent driving 
    interaction."""
    return -1 / n ** 2 * _n_to_Ngamma(n)


def _n_to_Ngamma(n):
    """Using the Clausius-Mossotti relation, convert a refractive index `n` into a (dimensionless) polarizability
    density `Ngamma`."""
    chi = n ** 2 - 1
    return 3 * chi / (3 + chi)


def _hw_alternate(value1, value2, total_num, value0=None, valuem=None):
    """Create an array that alternates like a half-wave stack.

    If `value0` or `valuem` are given, these will be the first and last entries in the array. The rest of the entries
    will alternate between `value1` and `value2`, starting with `value1`.

    (Mostly) optimized for speed.

    `value1`, `value2`, `value0`, and `valuem` can all have shape (x,), in which case the output will have shape
    (x, `total_num`).
    """
    # create an empty array with the right shape
    value_shape = np.broadcast(value1, value2, value0, valuem).shape
    alternating_list = np.empty((total_num,) + value_shape, np.complex128)  # shape (`total_num`, x)

    # fill the array
    if value0 is not None:
        value1, value2 = value2, value1
    alternating_list[::2] = value1
    alternating_list[1::2] = value2
    if valuem is not None:
        alternating_list[-1] = valuem
    if value0 is not None:
        alternating_list[0] = value0

    # adjust axis order
    alternating_list = np.moveaxis(alternating_list, 0, -1)                 # shape (x, `total_num`)
    return alternating_list


# TESTS


def _test_boost():
    # TODO: Include complex numbers in this test.
    n = [1, 2, 1]
    deltas = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])[:, np.newaxis] * np.pi
    A = [0, 1, 0]
    boosts = [boost(n, delta, A) for delta in deltas]

    correct_boosts = [0, 2 / np.sqrt(5), 1, 2 / np.sqrt(5), 0, 2 / np.sqrt(5), 1]
    assert np.allclose(boosts, correct_boosts)


def _test_boost_with_vectorization():
    # TODO: Include complex numbers in this test.
    n = np.array([[[1, 2, 1]] * 7, [[1, 3, 2]] * 7])
    deltas = np.array([[0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 0.25, 1, 1.5, 2, 2.75, 3]])[..., np.newaxis] * np.pi
    A = np.array([[[0, 1, 0]] * 7, [[0, 5, 0]] * 7])
    boosts = boost(n, deltas, A)

    correct_boosts = np.array([[0, 2 / np.sqrt(5), 1, 2 / np.sqrt(5), 0, 2 / np.sqrt(5), 1],
                              [0.0, 3.284689647114855, 6.666666666666668, 4.916660830178168, 1.2246467991473535e-15,
                              5.999415983240223, 6.666666666666668]])
    assert boosts.shape == correct_boosts.shape
    assert np.allclose(boosts, correct_boosts)


def _test_hw_alternate():
    value0 = np.arange(6).reshape(2, 3)
    valuem = np.arange(3) + 1000
    value1 = -1
    value2 = np.arange(6).reshape(2, 3) - 1000
    total_num1 = 7
    total_num2 = 2
    total_num3 = 4
    alternating_list1 = _hw_alternate(value1, value2, total_num1, value0, valuem)
    alternating_list2 = _hw_alternate(value1, value2, total_num2, value0, valuem)
    alternating_list3 = _hw_alternate(value1, value2, total_num3)

    correct_list1 = np.array([[[0, -1, -1000, -1, -1000, -1, 1000],
                               [1, -1, -999, -1, -999, -1, 1001],
                               [2, -1, -998, -1, -998, -1, 1002]],
                              [[3, -1, -997, -1, -997, -1, 1000],
                               [4, -1, -996, -1, -996, -1, 1001],
                               [5, -1, -995, -1, -995, -1, 1002]]])
    correct_list2 = np.array([[[0, 1000], [1, 1001], [2, 1002]], [[3, 1000], [4, 1001], [5, 1002]]])
    correct_list3 = np.array([[[-1, -1000, -1, -1000], [-1, -999, -1, -999], [-1, -998, -1, -998]],
                              [[-1, -997, -1, -997], [-1, -996, -1, -996], [-1, -995, -1, -995]]])
    assert alternating_list1.shape == correct_list1.shape
    assert alternating_list2.shape == correct_list2.shape
    assert alternating_list3.shape == correct_list3.shape
    assert np.allclose(alternating_list1, correct_list1)
    assert np.allclose(alternating_list2, correct_list2)
    assert np.allclose(alternating_list3, correct_list3)


def _time_hw_alternate():
    from timeit import repeat
    setup = '''
from __main__ import np, _hw_alternate
value0 = np.arange(10000).reshape(10, 1000)
value1 = np.ones(1000)
value2 = np.zeros((10, 1))
valuem = np.arange(10000).reshape(10, 1000)
total_num = 1000'''
    time = repeat('_hw_alternate(value1, value2, total_num, value0, valuem)', setup=setup, number=10)
    print(time)


if __name__ == '__main__':
    _test_boost_with_vectorization()
    _test_hw_alternate()
    _time_hw_alternate()

    # from matplotlib import pyplot as plt
    # A = [0, 1] * 20 + [0]
    # n = [1, 2, 1, 2, 1]
    # delta = [np.pi] * 3
    # b = boost(n, delta, A)
    # omega = np.linspace(0, 2, 10000)

    # b = hw_boost(omega, 1, lambda omega: 3, lambda omega: 1, A=A)
    # plt.plot(omega, b)
    # plt.xlabel(r'$\omega / \omega_0$')
    # plt.ylabel(r'boost factor $|B|$')
    # plt.gca().set_yscale('log')
    # plt.ylim(1e-1, 1e2)
    # plt.show()
