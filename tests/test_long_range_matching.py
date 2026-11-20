import numpy as np
import pytest

import coldscatter as cs
from coldscatter.asymptotics import match_k_matrix


def test_long_range_zero_coefficients_reproduce_free_reference():
    mu, energy, r1, r2 = 1.0, 0.02, 40.0, 40.1
    k2 = np.array([2 * mu * energy])
    M = np.array([[1.13]])
    free = match_k_matrix(M, r1, r2, k2, 0, mu=mu)
    zero = match_k_matrix(
        M, r1, r2, k2, 0, mu=mu,
        long_range=cs.LongRange({6: 0.0}, r_far=5000.0),
    )
    np.testing.assert_allclose(zero[0], free[0], rtol=2e-8, atol=2e-9)


def test_long_range_open_reference_is_matching_radius_stable():
    mu, c6, energy = 1.0, 1.0e5, 0.02
    wall = 6.5
    c12 = 0.5 * c6 * 8.0**6
    potential = cs.LennardJones(c6, c12)
    tail = cs.LongRange({6: c6}, r_far=5000.0)
    grid_results = []
    for rmax in (30.0, 50.0, 80.0):
        grid = cs.Grid(wall, rmax, 0.002)
        model = cs.CoupledChannels(
            mu, [0.0], [[potential]], long_range=tail
        )
        grid_results.append(model.solve(energy, grid).K[0, 0])
    assert np.ptp(grid_results) < 2e-5


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_closed_long_range_ratio_is_finite(ell):
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)
    ratio = _long_closed_ratio(tail, 1.0, ell, 0.05, 20.0, 20.1)
    assert np.isfinite(ratio)
    assert ratio > 0


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_threshold_long_range_ratio_is_finite(ell):
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)
    ratio = _long_closed_ratio(tail, 1.0, ell, 0.0, 20.0, 20.1)
    assert np.isfinite(ratio)
    assert ratio > 0


def test_long_range_can_be_assigned_per_channel():
    tail = cs.LongRange({6: 1000.0})
    model = cs.CoupledChannels(
        1.0, [0.0, 0.1],
        [[cs.Gaussian(0.1, 5.0, 1.0), 0.0], [None, cs.Gaussian(0.1, 5.0, 1.0)]],
        long_range=[tail, None],
    )
    assert model.long_range == (tail, None)


def test_long_range_validation():
    with pytest.raises(ValueError):
        cs.LongRange({2: 1.0})
    with pytest.raises(ValueError):
        cs.LongRange({6: np.inf})
    with pytest.raises(ValueError):
        cs.LongRange({6: 1.0}, r_far=0)
    with pytest.raises(ValueError):
        cs.CoupledChannels(
            1.0, [0.0, 0.1],
            [[0.0, 0.0], [None, 0.0]],
            long_range=[cs.LongRange({6: 1.0})],
        )


def test_long_range_is_reference_only_not_added_to_physical_potential():
    c6 = 4700.0
    pot = cs.LennardJones(c6, c6 * 11.0**6 / 2)
    tail = cs.LongRange({6: c6})
    model = cs.CoupledChannels(1.0, [0.0], [[pot]], long_range=tail)
    r = np.array([20.0, 30.0])
    np.testing.assert_allclose(model.potential_matrix(r)[:, 0, 0], pot(r))
    np.testing.assert_allclose(tail.potential(r), -c6 / r**6)


def test_long_range_scan_matches_independent_solves_and_preserves_unitarity():
    """Exercise long-range matching through batched and single-field solver paths."""
    coupling = cs.Gaussian(2.0e-4, 5.0, 1.0)
    diagonal = cs.Gaussian(1.0e-3, 5.0, 1.0)

    def thresholds(field):
        return np.array([0.0, 0.1 + 0.01 * field])

    tail = cs.LongRange({6: 10.0}, r_far=1000.0)
    model = cs.CoupledChannels(
        1.0,
        thresholds,
        [[diagonal, coupling], [None, diagonal]],
        long_range=tail,
    )
    grid = cs.Grid(2.0, 40.0, 0.05)
    fields = np.array([-20.0, 0.0, 20.0])

    scanned = model.scan(fields, 0.02, grid)
    with pytest.warns(
        UserWarning, match="classically allowed at r_min"
    ):
        independent = [
            model.solve(0.02, grid, B=float(field))
            for field in fields
        ]

    assert len(scanned) == len(fields)

    for batch_result, single_result in zip(scanned, independent):
        np.testing.assert_allclose(
            batch_result.K, single_result.K, rtol=2e-9, atol=2e-10
        )
        np.testing.assert_allclose(
            batch_result.S, single_result.S, rtol=2e-9, atol=2e-10
        )

        S = batch_result.S
        np.testing.assert_allclose(
            S @ S.conj().T,
            np.eye(len(batch_result.open_channels)),
            rtol=0.0,
            atol=2e-9,
        )
        np.testing.assert_allclose(S, S.T, rtol=0.0, atol=2e-9)

    assert len(scanned[0].open_channels) == 2
    assert len(scanned[1].open_channels) == 1
    assert len(scanned[2].open_channels) == 1


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_long_range_open_reference_preserves_wronskian(ell):
    """The open reference pair must preserve W(F,G) = -1."""
    from coldscatter.asymptotics import _long_open_values

    mu = 1.0
    energy = 0.02
    k = np.sqrt(2.0 * mu * energy)
    r1, r2 = 30.0, 30.1
    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)

    f1, fp1, g1, gp1, f2, fp2, g2, gp2 = _long_open_values(
        tail, mu, ell, k, r1, r2
    )

    w1 = f1 * gp1 - fp1 * g1
    w2 = f2 * gp2 - fp2 * g2

    np.testing.assert_allclose(
        [w1, w2], [-1.0, -1.0], rtol=0.0, atol=1.0e-9
    )


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_long_range_open_matching_converges_with_r_far(ell):
    """Matching K should be insensitive to a sufficiently distant start."""
    mu = 1.0
    energy = 0.02
    r1, r2 = 30.0, 30.1
    k2 = np.array([2.0 * mu * energy])
    M = np.array([[1.13]])
    r_fars = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]

    values = []
    for r_far in r_fars:
        tail = cs.LongRange({6: 1.0e5}, r_far=r_far)
        K, _, _, _ = match_k_matrix(
            M, r1, r2, k2, ell, mu=mu, long_range=tail
        )
        values.append(K[0, 0])

    relative_spread = np.ptp(values) / max(abs(v) for v in values)
    assert relative_spread < 1.0e-7


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
@pytest.mark.parametrize("kappa", [0.05, 0.0])
def test_closed_long_range_ratio_matches_direct_wavefunction(ell, kappa):
    """Compare piecewise wavefunction propagation with direct integration."""
    from scipy.integrate import solve_ivp
    from scipy.special import kve
    from coldscatter.asymptotics import _long_closed_ratio

    mu = 1.0
    r_far = 5000.0
    r1, r2 = 20.0, 20.1
    tail = cs.LongRange({6: 1.0e5}, r_far=r_far)

    if kappa == 0.0:
        y0 = -float(ell) / r_far
    else:
        x = kappa * r_far
        nu = ell + 0.5
        y0 = kappa * (
            0.5 / x
            - 0.5 * (
                kve(abs(nu - 1.0), x) + kve(nu + 1.0, x)
            ) / kve(nu, x)
        )

    def rhs(r, state):
        D, Dp = state
        # kappa**2 = -k2 already contains the factor 2 mu
        q = (
            ell * (ell + 1) / r**2
            + kappa**2
            + 2.0 * mu * tail.potential(r)
        )
        return Dp, q * D

    direct = solve_ivp(
        rhs,
        (r_far, r1),
        (1.0, y0),
        method="DOP853",
        t_eval=(r2, r1),
        rtol=2e-11,
        atol=2e-13,
    )
    assert direct.success

    D2, D1 = direct.y[0]
    direct_ratio = D1 / D2

    propagated_ratio = _long_closed_ratio(
        tail, mu, ell, kappa, r1, r2
    )

    assert np.isfinite(direct_ratio)
    np.testing.assert_allclose(
        propagated_ratio, direct_ratio, rtol=2e-9, atol=2e-11
    )



def test_closed_long_range_ratio_crosses_a_wavefunction_node():
    """The signed ratio agrees with direct integration across a node."""
    from scipy.integrate import solve_ivp
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e6}, r_far=5000.0)
    mu, ell, kappa = 1.0, 0, 0.0
    r1, r2 = 10.9, 11.0

    def rhs(r, state):
        D, Dp = state
        # kappa**2 = -k2 already contains the factor 2 mu
        q = (
            ell * (ell + 1) / r**2
            + kappa**2
            + 2.0 * mu * tail.potential(r)
        )
        return Dp, q * D

    direct = solve_ivp(
        rhs,
        (tail.r_far, r1),
        (1.0, 0.0),
        method="DOP853",
        t_eval=(r2, r1),
        rtol=2e-12,
        atol=2e-14,
        max_step=0.02,
    )
    assert direct.success

    direct_ratio = direct.y[0, 1] / direct.y[0, 0]
    ratio = _long_closed_ratio(tail, mu, ell, kappa, r1, r2)

    assert np.isfinite(ratio)
    assert direct_ratio < 0.0
    assert ratio < 0.0
    np.testing.assert_allclose(
        ratio, direct_ratio, rtol=3e-9, atol=2e-11
    )


@pytest.mark.parametrize(
    "ell,kappa",
    [(0, 0.05), (1, 0.05), (0, 0.0)],
)
def test_closed_long_range_ratio_converges_with_r_far(ell, kappa):
    """The closed-channel reference ratio is stable as r_far increases."""
    from coldscatter.asymptotics import _long_closed_ratio

    radii = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    values = [
        _long_closed_ratio(
            cs.LongRange({6: 1.0e5}, r_far=r_far),
            1.0, ell, kappa, 20.0, 20.1,
        )
        for r_far in radii
    ]

    reference = values[-1]
    np.testing.assert_allclose(
        values, reference, rtol=3e-10, atol=2e-12
    )


def test_closed_long_range_ratio_rejects_invalid_matching_radii():
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)

    with pytest.raises(ValueError):
        _long_closed_ratio(tail, 1.0, 0, 0.05, 20.1, 20.0)

    with pytest.raises(ValueError):
        _long_closed_ratio(tail, 1.0, 0, 0.05, 20.0, 5000.0)


def test_long_range_coefficients_are_immutable_snapshots():
    source = {6: 1000.0, 8: 20.0}
    tail = cs.LongRange(source)

    source[6] = 9999.0
    assert tail.coefficients[6] == 1000.0
    assert tail.terms == ((6, 1000.0), (8, 20.0))

    with pytest.raises(TypeError):
        tail.coefficients[6] = 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coefficients": {6: True}},
        {"coefficients": {6: "1000"}},
        {"coefficients": {True: 1000.0}},
        {"coefficients": {6: 1000.0}, "r_far": True},
        {"coefficients": {6: 1000.0}, "rtol": True},
        {"coefficients": {6: 1000.0}, "atol": True},
    ],
)
def test_long_range_rejects_boolean_and_non_real_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        cs.LongRange(**kwargs)


def test_long_range_potential_rejects_floating_point_overflow():
    tail = cs.LongRange({6: 1.0e300})
    with pytest.raises(FloatingPointError):
        tail.potential(1.0e-100)


def test_long_range_open_reference_rejects_invalid_radii_and_parameters():
    from coldscatter.asymptotics import _long_open_values

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)

    with pytest.raises(ValueError):
        _long_open_values(tail, 1.0, 0, 0.1, 30.1, 30.0)

    with pytest.raises(ValueError):
        _long_open_values(tail, 1.0, 0, 0.1, 30.0, 5000.0)

    with pytest.raises(ValueError):
        _long_open_values(tail, 1.0, 0, 0.0, 30.0, 30.1)

    with pytest.raises(ValueError):
        _long_open_values(tail, 1.0, -1, 0.1, 30.0, 30.1)


def test_coupled_channels_rejects_boolean_ell():
    potential = cs.Constant(0.0)
    with pytest.raises(ValueError, match="ell"):
        cs.CoupledChannels(
            mu=1.0, thresholds=[0.0],
            potentials=[[potential]], ell=True,
        )
    with pytest.raises(ValueError, match="ell"):
        cs.CoupledChannels(
            mu=1.0, thresholds=[0.0],
            potentials=[[potential]], ell=[0, True],
        )


@pytest.mark.parametrize("ell", [True, -1, 1.5, float("nan"), [0, 1]])
def test_match_k_matrix_rejects_invalid_ell(ell):
    from coldscatter.asymptotics import match_k_matrix

    with pytest.raises(ValueError, match="ell"):
        match_k_matrix(
            np.eye(1), 10.0, 10.1, np.array([0.1]), ell
        )


@pytest.mark.parametrize("ell", [True, -1, 1.5])
def test_long_closed_ratio_rejects_invalid_ell(ell):
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)
    with pytest.raises(ValueError, match="ell"):
        _long_closed_ratio(tail, 1.0, ell, 0.05, 20.0, 20.1)


@pytest.mark.parametrize(
    "r1,r2",
    [(float("nan"), 20.1), (20.0, float("inf")), (0.0, 20.1)],
)
def test_long_closed_ratio_rejects_nonfinite_or_nonpositive_radii(r1, r2):
    from coldscatter.asymptotics import _long_closed_ratio

    tail = cs.LongRange({6: 1.0e5}, r_far=5000.0)
    with pytest.raises(ValueError, match="radii"):
        _long_closed_ratio(tail, 1.0, 0, 0.05, r1, r2)
