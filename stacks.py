"""Using transfer matrices to describe 1D setups with distributed radiation sources.

TODO: Double-check that Millar's formalism works for complex refractive indices.
"""
import numpy as np


def boost(n, delta, A):
    """General transfer matrix formalism.
    `n`, `delta`, and `A` are arrays, containing values for each region in the stack.
    Uses the convention that a positive imaginary part of the refractive index corresponds to absorption (same
    convention that Millar uses).

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

    Tmm = np.broadcast_to(np.eye(2), n.shape[:-1] + (2, 2))     # shape (x, 2, 2)
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


def _hw_boost(omega, omega0, n1, n2, n1_real0, n2_real0, n0, nm, num_layers):
    """Does the work for `hw_boost`. The induced electric field `A` is calculated automatically by `_A`.

    :param omega: angular frequency
    :param omega0: resonant angular frequency of the half-wave stack
    :param n1: complex refractive index of layer 1 (for the given omega)
    :param n2: complex refractive index of layer 2 (for the given omega)
    :param n1_real0: real refractive index of layer 1 on resonance (sets width of layer 1)
    :param n2_real0: real refractive index of layer 2 on resonance (sets width of layer 2)
    :param n0: complex refractive index of "left" outer (infinite) region (for the given omega)
    :param nm: complex refractive index of "right" outer (infinite) region (for the given omega)
    :param num_layers: number of (finite) layers in the stack
    :return: boost factor
    """
    delta1 = _hw_delta(n1, n1_real0, omega, omega0)
    delta2 = _hw_delta(n2, n2_real0, omega, omega0)
    delta = _hw_alternate(delta2, delta1, num_layers)

    n = _hw_alternate(n1, n2, num_layers + 2, n0, nm)
    A = _A(n)
    return boost(n, delta, A)


def hw_boost(omega, omega0, n1, n2, num_layers, n1_real0=None, n2_real0=None, n0=1, nm=1):
    """Boost factor for a half-wave stack with resonant frequency `omega0`.

    :param omega: angular frequency, can be a 1D array
    :param omega0: resonant angular frequency of the half-wave stack
    :param n1: complex refractive index of layer 1, can be a function of omega or a constant
    :param n2: complex refractive index of layer 2, can be a function of omega or a constant
    :param num_layers: Number of (finite size) layers in the stack.
    :param n1_real0: real refractive index of layer 1 on resonance (sets width of layer 1)
    :param n2_real0: real refractive index of layer 2 on resonance (sets width of layer 2)
    :param n0: complex refractive index of "left" outer (infinite) region, can be a function of omega or a constant
    :param nm: complex refractive index of "right" outer (infinite) region, can be a function of omega or a constant
    :return: boost factor
    """
    # turn n's into functions
    if not callable(n1):
        n1_ = n1
        def n1(omega):
            return n1_
    if not callable(n2):
        n2_ = n2
        def n2(omega):
            return n2_
    if not callable(n0):
        n0_ = n0
        def n0(omega):
            return n0_
    if not callable(nm):
        nm_ = nm
        def nm(omega):
            return nm_

    if not n1_real0:
        n1_real0 = np.real(n1(omega0))
    if not n2_real0:
        n2_real0 = np.real(n2(omega0))

    try:  # iterable omega
        result = np.array([_hw_boost(
            omega_, omega0, n1(omega_), n2(omega_), n1_real0, n2_real0, n0(omega), nm(omega), num_layers
        ) for omega_ in omega])
    except TypeError:
        result = _hw_boost(omega, omega0, n1(omega), n2(omega), n1_real0, n2_real0, n0(omega), nm(omega), num_layers)
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

    Entries along the last axis will alternate between `value1` and `value2`, with `value1` for the odd indices. If
    `value0` or `valuem` are given, they will override the first and last entries, respectively.

    (Mostly) optimized for speed.

    `value1`, `value2`, `value0`, and `valuem` can have shape (x,), and the output will have shape (x, `total_num`).
    """
    value_shape = np.broadcast(value1, value2, value0, valuem).shape
    alternating_list = np.empty((total_num,) + value_shape, np.complex128)  # shape (`total_num`, x)
    alternating_list[1::2] = value1
    alternating_list[::2] = value2
    if valuem is not None:
        alternating_list[-1] = valuem
    if value0 is not None:
        alternating_list[0] = value0
    alternating_list = np.moveaxis(alternating_list, 0, -1)  # shape (x, `total_num`)
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
    alternating_list3 = _hw_alternate(value2, value1, total_num3)

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
    from __main__ import np, _hw_alternate, _hw_alternate2
    value0 = np.arange(10000).reshape(10, 1000)
    value1 = np.ones(1000)
    value2 = np.zeros((10, 1))
    valuem = np.arange(10000).reshape(10, 1000)
    total_num = 1000'''
    time = repeat('_hw_alternate(value0, value1, value2, valuem, total_num)', setup=setup, number=10)
    print(time)


if __name__ == '__main__':
    _test_boost_with_vectorization()
    _test_hw_alternate()

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
