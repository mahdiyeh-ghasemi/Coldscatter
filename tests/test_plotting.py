import csv

import numpy as np
import pytest

import coldscatter as cs
from coldscatter import units as u
from coldscatter.plotting import export_results_csv, plot_cross_sections, plot_scattering_length


@pytest.fixture(scope="module")
def scan_results():
    mu = u.reduced_mass(87.0, 87.0)
    c6 = 4700.0
    model = cs.two_channel_feshbach(
        mu, c6,
        c12_open=c6 * 11.0**6 / 2,
        c12_closed=c6 * 10.4**6 / 2,
        coupling=2e-4,
        coupling_r0=11.0,
        coupling_sigma=1.5,
        delta_mu=2 * u.MU_B_HARTREE_PER_GAUSS,
    )
    fields = np.array([5.0, 10.0, 15.0])
    return model.scan(fields, u.from_kelvin(1e-7), cs.Grid(7.0, 800.0, 0.02))


def test_csv_export_contains_observables(tmp_path, scan_results):
    path = export_results_csv(scan_results, tmp_path / "data" / "scan.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == len(scan_results)
    assert float(rows[0]["field_G"]) == scan_results[0].field
    assert rows[0]["scattering_length_real_bohr"] != ""
    assert "elastic" not in rows[0]  # explicit unit-bearing column names
    assert float(rows[0]["cross_section_total_bohr2"]) >= 0.0


def test_csv_rejects_empty_results(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        export_results_csv([], tmp_path / "empty.csv")


def test_plot_functions_return_figure_and_axes(tmp_path, scan_results):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)

    fig1, ax1 = plot_scattering_length(
        scan_results, x="field", save=tmp_path / "plots" / "a.png"
    )
    fig2, ax2 = plot_cross_sections(
        scan_results, x="field", save=tmp_path / "plots" / "cross.png"
    )

    assert len(ax1.lines) == 3  # two data curves and the zero reference line
    assert len(ax2.lines) == 5  # four data curves and the zero reference line
    assert (tmp_path / "plots" / "a.png").is_file()
    assert (tmp_path / "plots" / "cross.png").is_file()

    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(fig2)


def test_plot_rejects_invalid_x(scan_results):
    with pytest.raises(ValueError, match="x must"):
        plot_scattering_length(scan_results, x="temperature")
