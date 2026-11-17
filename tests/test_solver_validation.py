import numpy as np
import pytest

from coldscatter import Constant, CoupledChannels


def test_upper_triangle_coupling_is_symmetrized():
    coupling = Constant(0.25)
    problem = CoupledChannels(
        mu=1.0,
        thresholds=[0.0, 0.0],
        potentials=[
            [Constant(0.0), coupling],
            [None, Constant(0.0)],
        ],
    )

    matrix = problem.potential_matrix(np.array([1.0, 2.0]))
    np.testing.assert_allclose(matrix[:, 0, 1], 0.25)
    np.testing.assert_allclose(matrix[:, 1, 0], 0.25)


@pytest.mark.parametrize(
    "potentials",
    [
        [
            [Constant(0.0), None],
            [Constant(0.25), Constant(0.0)],
        ],
        [
            [Constant(0.0), Constant(0.25)],
            [Constant(0.25), Constant(0.0)],
        ],
    ],
)
def test_lower_triangle_entries_are_rejected(potentials):
    with pytest.raises(ValueError, match="Only the upper triangle is accepted"):
        CoupledChannels(
            mu=1.0,
            thresholds=[0.0, 0.0],
            potentials=potentials,
        )
