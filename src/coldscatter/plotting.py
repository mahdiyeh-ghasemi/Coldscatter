"""Publication-oriented plotting and tabular export for Coldscatter results.

Matplotlib is imported lazily: it is an optional dependency and is not
required to import or use the numerical solver.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np


def _validated_results(results, x):
    if x not in {"field", "energy"}:
        raise ValueError("x must be 'field' or 'energy'")
    items = list(results)
    if not items:
        raise ValueError("results must contain at least one ScatteringResult")
    if not all(hasattr(item, "cross_sections") and hasattr(item, x) for item in items):
        raise TypeError("results must contain ScatteringResult-like objects")
    values = np.asarray([getattr(item, x) for item in items], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"all {x} values must be finite")
    return items, values


def _json_array(values):
    """Serialize real or complex NumPy arrays without losing information."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        payload = {
            "real": array.real.tolist(),
            "imag": array.imag.tolist(),
        }
    else:
        payload = array.tolist()
    return json.dumps(payload, separators=(",", ":"))


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires Matplotlib. Install it with "
            "`python -m pip install 'coldscatter[examples]'`."
        ) from exc
    return plt


def plot_scattering_length(results, *, x="field", ax=None, save=None):
    """Plot real and imaginary entrance-channel s-wave scattering lengths.

    Parameters
    ----------
    results : sequence of ScatteringResult
        Results from ``CoupledChannels.scan`` or individual solves.
    x : {'field', 'energy'}
        Horizontal variable. Field is in gauss; energy is in hartree.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure and axes are created.
    save : path-like, optional
        If supplied, save the figure to this path.

    Returns
    -------
    (figure, axes)
        Matplotlib figure and axes objects.
    """
    items, values = _validated_results(results, x)
    lengths = np.asarray([item.scattering_length() for item in items], dtype=complex)

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
    else:
        fig = ax.figure

    ax.plot(values, lengths.real, label=r"$\mathrm{Re}\,a$", linewidth=1.6)
    ax.plot(values, lengths.imag, label=r"$\mathrm{Im}\,a$", linewidth=1.6)
    xlabel = "Magnetic field (G)" if x == "field" else "Collision energy (hartree)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Scattering length ($a_0$)")
    ax.axhline(0.0, linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()

    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_cross_sections(results, *, x="field", ax=None, save=None):
    """Plot elastic, inelastic, loss, and total cross sections.

    Cross sections are shown in bohr squared. The linear vertical scale is
    intentional: it preserves small or slightly negative numerical loss
    values instead of silently hiding them on a logarithmic axis.
    """
    items, values = _validated_results(results, x)
    data = [item.cross_sections() for item in items]
    keys = ("elastic", "inelastic", "loss", "total")

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
    else:
        fig = ax.figure

    for key in keys:
        y = np.asarray([entry[key] for entry in data], dtype=float)
        ax.plot(values, y, label=key.capitalize(), linewidth=1.5)

    xlabel = "Magnetic field (G)" if x == "field" else "Collision energy (hartree)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cross section ($a_0^2$)")
    ax.axhline(0.0, linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()

    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def export_results_csv(results, path):
    """Export observables and channel-resolved matrices to a CSV file.

    Arrays and channel-indexed dictionaries are stored as compact JSON
    strings in individual CSV cells. Scattering length is left blank for
    non-s-wave entrance channels.
    """
    items = list(results)
    if not items:
        raise ValueError("results must contain at least one ScatteringResult")

    columns = [
        "field_G", "energy_hartree", "mu_au", "entrance_channel",
        "open_channels", "k_open_au", "ell",
        "scattering_length_real_bohr", "scattering_length_imag_bohr",
        "K_matrix", "S_matrix", "k_asymmetry",
        "cross_section_elastic_bohr2", "cross_section_inelastic_bohr2",
        "cross_section_loss_bohr2", "cross_section_total_bohr2",
        "cross_section_to_channel_bohr2",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in items:
            try:
                a = item.scattering_length()
                a_real, a_imag = float(a.real), float(a.imag)
            except ValueError:
                a_real, a_imag = "", ""

            cross = item.cross_sections()
            writer.writerow({
                "field_G": item.field,
                "energy_hartree": item.energy,
                "mu_au": item.mu,
                "entrance_channel": item.entrance,
                "open_channels": json.dumps(np.asarray(item.open_channels).tolist()),
                "k_open_au": json.dumps(np.asarray(item.k_open).tolist()),
                "ell": json.dumps(np.asarray(item.ell).tolist()),
                "scattering_length_real_bohr": a_real,
                "scattering_length_imag_bohr": a_imag,
                "K_matrix": _json_array(item.K),
                "S_matrix": _json_array(item.S),
                "k_asymmetry": item.k_asymmetry,
                "cross_section_elastic_bohr2": cross["elastic"],
                "cross_section_inelastic_bohr2": cross["inelastic"],
                "cross_section_loss_bohr2": cross["loss"],
                "cross_section_total_bohr2": cross["total"],
                "cross_section_to_channel_bohr2": json.dumps(
                    cross["to_channel"], sort_keys=True
                ),
            })
    return output
