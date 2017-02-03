import matplotlib.pyplot as plt
import numpy as np

from tbplot.structure import (plot_sites, plot_hoppings, plot_periodic_boundaries,
                              structure_plot_properties)
from . import pltutils

__all__ = ["plot_system", "plot_lead", "plot_system_with_leads"]


def _center(pos, shift):
    """Return the 2D center position of `pos + shift`"""
    x = np.concatenate((pos[0], pos[0] + shift[0]))
    y = np.concatenate((pos[1], pos[1] + shift[1]))
    return (x.max() + x.min()) / 2, (y.max() + y.min()) / 2


def _decorate_structure_plot(axes="xy", add_margin=True, **_):
    plt.gca().set_aspect("equal")
    plt.xlabel("{}".format(axes[0]))
    plt.ylabel("{}".format(axes[1]))
    if add_margin:
        pltutils.set_min_axis_length(0.5)
        pltutils.set_min_axis_ratio(0.4)
        pltutils.despine(trim=True)
        pltutils.add_margin()
    else:
        pltutils.despine()


def plot_system(smap, num_periods=1, **kwargs):
    """Plot the structure: sites, hoppings and periodic boundaries (if any)

    Parameters
    ----------
    smap : StructureMap
    num_periods : int
        Number of times to repeat the periodic boundaries.
    **kwargs
        Additional plot arguments as specified in :func:`.structure_plot_properties`.
    """
    props = structure_plot_properties(**kwargs)

    plot_hoppings(smap.positions, smap.hoppings, **props['hopping'])
    plot_sites(smap.positions, smap.sublattices, **props['site'])
    plot_periodic_boundaries(smap.positions, smap.hoppings, smap.boundaries, smap.sublattices,
                             num_periods, **props)

    _decorate_structure_plot(**props)


def plot_lead(lead_smap, index, lead_length=6, **kwargs):
    """Plot the sites, hoppings and periodic boundaries of the lead

    Parameters
    ----------
    lead_smap : StructureMap
    index : int
        This number will appear on the lead label.
    lead_length : int
        Number of times to repeat the lead's periodic boundaries.
    **kwargs
        Additional plot arguments as specified in :func:`.structure_plot_properties`.
    """
    pos = lead_smap.positions
    sub = lead_smap.sublattices
    inner_hoppings = lead_smap.hoppings.tocoo()
    boundary = lead_smap.boundaries[0]
    outer_hoppings = boundary.hoppings.tocoo()

    props = structure_plot_properties(**kwargs)

    blend_gradient = np.linspace(0.5, 0.1, lead_length)
    for i, blend in enumerate(blend_gradient):
        offset = i * boundary.shift
        plot_sites(pos, sub, offset=offset, blend=blend, **props['site'])
        plot_hoppings(pos, inner_hoppings, offset=offset, blend=blend, **props['hopping'])
        plot_hoppings(pos, outer_hoppings, offset=offset - boundary.shift, blend=blend,
                      boundary=(1, boundary.shift), **props['boundary'])

    label_pos = _center(pos, lead_length * boundary.shift * 1.5)
    pltutils.annotate_box("lead {}".format(index), label_pos, bbox=dict(alpha=0.7))

    _decorate_structure_plot(**props)


def plot_system_with_leads(smap, leads, num_periods=1, lead_length=6, axes='xy', **kwargs):
    """Plot the structure of the model: sites, hoppings, boundaries and leads

    Parameters
    ----------
    smap : StructureMap
    leads : List[StructureMap]
    num_periods : int
        Number of times to repeat the periodic boundaries.
    lead_length : int
        Number of times to repeat the lead structure.
    axes : str
        The spatial axes to plot. E.g. 'xy', 'yz', etc.
    **kwargs
        Additional plot arguments as specified in :func:`.structure_plot_properties`.
    """
    kwargs['add_margin'] = False
    plot_system(smap, num_periods, axes=axes, **kwargs)
    for n, lead_smap in enumerate(leads):
        plot_lead(lead_smap, n, lead_length, axes=axes, **kwargs)
    _decorate_structure_plot(axes=axes)
