import pytest
import tbplot

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

shape = (5, 5)


@pytest.fixture
def sites():
    rows, cols = shape
    x = np.tile(np.linspace(0, 4, cols), rows)
    y = np.concatenate([np.linspace(2, 0, cols) + offset
                        for offset in np.linspace(0, -4, rows)])
    z = np.zeros(rows * cols)
    data = np.concatenate([np.roll(np.arange(cols), shift)
                           for shift in np.arange(rows)])

    return (x, y, z), data


@pytest.fixture
def hoppings(sites):
    rows, cols = shape
    size = rows * cols
    positions, _ = sites

    from_idx = np.tile(np.arange(size), 2)
    to_base = np.arange(size)
    to_idx = np.concatenate([to_base + 1, to_base + cols])
    to_idx[cols-1:size:cols] = -1

    keep = np.logical_and(to_idx >= 0, to_idx < rows * cols)
    from_idx, to_idx = (v[keep] for v in (from_idx, to_idx))

    return positions, scipy.sparse.coo_matrix((np.zeros(from_idx.size), (from_idx, to_idx)))


def test_plot_sites(assert_figure, sites):
    positions, data = sites

    with assert_figure():
        tbplot.plot_sites(positions, data, radius=0.2)
        plt.axis("equal")


def test_plot_hoppings(assert_figure, hoppings):
    positions, graph = hoppings

    with assert_figure():
        tbplot.plot_hoppings(positions, graph, width=1)
        plt.axis("equal")


def test_plot_sites_and_hoppings(assert_figure, sites, hoppings):
    positions, data = sites
    _, graph = hoppings

    with assert_figure():
        tbplot.plot_sites(positions, data, radius=0.2)
        tbplot.plot_hoppings(positions, graph, width=1)
        plt.axis("equal")
