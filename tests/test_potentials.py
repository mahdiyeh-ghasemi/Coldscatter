import numpy as np
import pytest
from math import gamma

import coldscatter as cs
from coldscatter.potentials import as_potential


def test_constant_potential_scalar_and_array():
    p = cs.Constant(-0.25)
    assert p(2.0) == pytest.approx(-0.25)
    np.testing.assert_allclose(p(np.array([1., 2., 3.])), -0.25)


def test_lennard_jones_formula():
    c6, c12 = 12.0, 4.0
    r = np.array([1., 2., 3.])
    np.testing.assert_allclose(
        cs.LennardJones(c6, c12)(r), c12 / r**12 - c6 / r**6
    )


def test_lennard_jones_from_depth():
    depth, r_min = 0.02, 2.5
    p = cs.LennardJones.from_depth(depth, r_min)
    assert p(r_min) == pytest.approx(-depth)
    assert p(r_min / 2**(1 / 6)) == pytest.approx(0.0, abs=1e-14)


def test_gaussian_formula():
    p = cs.Gaussian(-2., 3., 0.5)
    r = np.array([2., 3., 4.])
    np.testing.assert_allclose(
        p(r), -2. * np.exp(-0.5 * ((r - 3.) / 0.5)**2)
    )


def test_exponential_formula():
    p = cs.Exponential(2., 0.5)
    r = np.array([0., 1., 2.])
    np.testing.assert_allclose(p(r), 2. * np.exp(-0.5 * r))


def test_from_function_vectorization():
    p = cs.FromFunction(lambda r: np.where(r < 2., -1., 0.))
    np.testing.assert_allclose(p([1., 2., 3.]), [-1., 0., 0.])


def test_as_potential_conversions():
    assert as_potential(2.)(1.) == pytest.approx(2.)
    assert as_potential(lambda r: r**2)(2.) == pytest.approx(4.)
    assert as_potential(cs.Constant(3.))(1.) == pytest.approx(3.)


def test_potential_addition():
    p, q = cs.Constant(2.), cs.Constant(3.)
    np.testing.assert_allclose((p + q)([1., 2.]), [5., 5.])
    np.testing.assert_allclose((2. + p)([1., 2.]), [4., 4.])


def test_potential_scaling():
    p = cs.Gaussian(2., 0., 1.)
    np.testing.assert_allclose((3. * p)([0., 1.]), 3. * p([0., 1.]))
    np.testing.assert_allclose((p * 3.)([0., 1.]), 3. * p([0., 1.]))


def test_vdw_length_and_mean_scattering_length():
    mu, c6 = 1000., 4700.
    beta6 = cs.vdw_length(mu, c6)
    abar = cs.mean_scattering_length(mu, c6)
    assert beta6 > 0
    assert abar == pytest.approx(4. * np.pi / gamma(0.25)**2 * beta6)


@pytest.mark.parametrize("sigma", [0., -1.])
def test_gaussian_rejects_nonpositive_sigma(sigma):
    with pytest.raises(ValueError):
        cs.Gaussian(1., 0., sigma)


@pytest.mark.parametrize("mu,c6", [
    (0., 10.), (-1., 10.), (1., 0.)
])
def test_vdw_length_rejects_nonpositive_parameters(mu, c6):
    with pytest.raises(ValueError):
        cs.vdw_length(mu, c6)


@pytest.mark.parametrize("mu,c6", [
    (0., 10.), (-1., 10.), (1., 0.)
])
def test_mean_scattering_length_rejects_nonpositive_parameters(mu, c6):
    with pytest.raises(ValueError):
        cs.mean_scattering_length(mu, c6)


@pytest.mark.parametrize("r", [0., -1., np.inf, np.nan])
def test_lennard_jones_rejects_invalid_separation(r):
    with pytest.raises(ValueError):
        cs.LennardJones(10., 100.)(r)


@pytest.mark.parametrize("depth,r_min", [
    (0., 2.), (-1., 2.), (1., 0.), (1., -2.)
])
def test_lennard_jones_from_depth_rejects_invalid_parameters(depth, r_min):
    with pytest.raises(ValueError):
        cs.LennardJones.from_depth(depth, r_min)
