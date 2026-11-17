import numpy as np
import pytest

from coldscatter import Grid


def test_grid_exact_endpoint():
    grid = Grid(1.0, 2.0, 0.25)
    assert grid.n_points == 5
    assert grid.r[-1] == pytest.approx(2.0)


def test_grid_warns_when_step_does_not_reach_r_max():
    with pytest.warns(UserWarning, match="not a multiple of step"):
        grid = Grid(1.0, 2.0, 0.3)

    assert grid.n_points == 4
    np.testing.assert_allclose(grid.r, [1.0, 1.3, 1.6, 1.9])
    assert grid.r[-1] != pytest.approx(2.0)


@pytest.mark.parametrize(
    "r_min,r_max,step",
    [
        (0.0, 2.0, 0.1),
        (-1.0, 2.0, 0.1),
        (1.0, 2.0, 0.0),
        (1.0, 2.0, -0.1),
        (1.0, 1.0, 0.1),
        (np.nan, 2.0, 0.1),
        (1.0, np.inf, 0.1),
    ],
)
def test_grid_rejects_invalid_parameters(r_min, r_max, step):
    with pytest.raises(ValueError):
        Grid(r_min, r_max, step)
