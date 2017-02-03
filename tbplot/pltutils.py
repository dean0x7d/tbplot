"""Collection of utility functions for matplotlib"""
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .detail.utils import with_defaults

__all__ = ["axes", "backend", "despine", "respine", "set_min_axis_length", "set_min_axis_ratio",
           "add_margin", "blend_colors", "colorbar", "annotate_box", "cm2inch", "legend",
           "get_palette", "set_palette", "direct_cmap_norm", "align"]


@contextmanager
def axes(ax):
    """A context manager that sets the active Axes instance to `ax`

    Parameters
    ----------
    ax : plt.Axes

    Examples
    --------
    >>> f, (ax1, ax2) = plt.subplots(1, 2)
    >>> ax2 == plt.gca()
    True
    >>> with axes(ax1):
    ...    ax1 == plt.gca()
    True
    >>> ax2 == plt.gca()
    True
    """
    previous_ax = plt.gca()
    plt.sca(ax)
    yield
    plt.sca(previous_ax)


@contextmanager
def backend(new_backend):
    """Change the backend within this context

    Parameters
    ----------
    new_backend : str
        Name of a matplotlib backend.
    """
    old_backend = mpl.get_backend()
    plt.switch_backend(new_backend)
    try:
        yield
    finally:
        plt.switch_backend(old_backend)


def despine(trim=False):
    """Remove the top and right spines

    Parameters
    ----------
    trim : bool
        Trim spines so that they don't extend beyond the last major ticks.
    """
    ax = plt.gca()
    if ax.name == "3d":
        return

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    if trim:
        for v, side in [("x", "bottom"), ("y", "left")]:
            ax.spines[side].set_smart_bounds(True)
            ticks = getattr(ax, "get_{}ticks".format(v))()
            vmin, vmax = getattr(ax, "get_{}lim".format(v))()
            ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
            getattr(ax, "set_{}ticks".format(v))(ticks)


def respine():
    """Redraw all spines, opposite of :func:`despine`"""
    ax = plt.gca()
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_smart_bounds(False)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


def set_min_axis_length(length, axis="xy"):
    """Set minimum axis length

    Parameters
    ----------
    length : float
        Minimum range in data coordinates
    axis : {"x", "y", "xy"}
        Apply to a single axis ("x", "y") or both ("xy").
    """
    ax = plt.gca()
    for a in axis:
        _min, _max = getattr(ax, "get_{}lim".format(a))()
        if abs(_max - _min) < length:
            center = (_max + _min) / 2
            _min = center - length / 2
            _max = center + length / 2
            getattr(ax, "set_{}lim".format(a))(_min, _max, auto=None)


def set_min_axis_ratio(ratio):
    """Set minimum ratio between axes limits

    Parameters
    ----------
    ratio : float
    """
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x = (xmax - xmin) / 2
    y = (ymax - ymin) / 2

    if y != 0 and x / y < ratio:
        center = (xmax + xmin) / 2
        lim = ratio * y
        plt.xlim(center - lim, center + lim)
    elif y / x < ratio:
        center = (ymax + ymin) / 2
        lim = ratio * x
        plt.ylim(center - lim, center + lim)


def add_margin(margin=0.08, axis="xy"):
    """Adjust the axis length to include a margin (after autoscale)

    Parameters
    ----------
    margin : float
        Fraction of the original length.
    axis : {"x", "y", "xy"}
        Apply to a single axis ("x", "y") or both ("xy").
    """
    ax = plt.gca()
    for a in axis:
        _min, _max = getattr(ax, "get_{}lim".format(a))()
        set_min_axis_length(abs(_max - _min) * (1 + margin), axis=a)


def blend_colors(color, bg, factor):
    """Blend color with background

    Parameters
    ----------
    color
        Color that will be blended.
    bg
        Background color.
    factor : float
        Blend factor: 0 to 1.
    """
    from matplotlib.colors import colorConverter
    color, bg = (np.array(colorConverter.to_rgb(c)) for c in (color, bg))
    return (1 - factor) * bg + factor * color


def colorbar(mappable=None, cax=None, ax=None, label="", powerlimits=(0, 0), **kwargs):
    """Custom colorbar with modified style and optional label

    Changes default `pad` and `aspect` argument values and turns on rasterization for a
    nicer looking colorbar with smaller size in vector formats (pdf, svg).

    Parameters
    ----------
    label : str
        Color data label.
    powerlimits : Tuple[int, int]
        Sets size thresholds for scientific notation.
    mappable, cax, ax, **kwargs
        Forwarded to :func:`matplotlib.pyplot.colorbar`.
    """
    cbar = plt.colorbar(mappable, cax, ax, **with_defaults(kwargs, pad=0.02, aspect=28))

    cbar.solids.set_edgecolor("face")  # remove white gaps between segments
    cbar.solids.set_rasterized(True)  # and reduce pdf and svg output size

    if powerlimits and hasattr(cbar.formatter, "set_powerlimits"):
        cbar.formatter.set_powerlimits(powerlimits)
    cbar.update_ticks()

    if label:
        if cbar.formatter.get_offset() or cbar.orientation != "vertical":
            cbar.set_label(label)
        else:
            cbar.ax.set_xlabel(label)
            cbar.ax.xaxis.set_label_position("top")

    return cbar


def annotate_box(s, xy, fontcolor="black", **kwargs):
    """Annotate with a box around the text

    Parameters
    ----------
    s : str
        Text string.
    xy : Tuple[float, float]
        Text position.
    fontcolor : color
        Setting "white" will make the background black.
    **kwargs
        Forwarded to `plt.annotate()`.
    """
    kwargs["bbox"] = with_defaults(
        kwargs.get("bbox", {}),
        boxstyle="round,pad=0.2", alpha=0.5, lw=0.3,
        fc="white" if fontcolor != "white" else "black"
    )

    if all(key in kwargs for key in ["arrowprops", "xytext"]):
        kwargs["arrowprops"] = with_defaults(
            kwargs["arrowprops"], dict(arrowstyle="->", color=fontcolor)
        )

    plt.annotate(s, xy, **with_defaults(kwargs, color=fontcolor, horizontalalignment="center",
                                        verticalalignment="center"))


def cm2inch(*values):
    """Convert from centimeter to inch

    Parameters
    ----------
    *values

    Returns
    -------
    tuple

    Examples
    --------
    >>> cm2inch(2.54, 5.08)
    (1.0, 2.0)
    """
    return tuple(v / 2.54 for v in values)


def legend(*args, reverse=False, facecolor="0.98", lw=0, **kwargs):
    """Custom legend with modified style and option to reverse label order

    Parameters
    ----------
    reverse : bool
        Reverse the label order.
    facecolor : color
        Legend background color.
    lw : float
        Frame width.
    *args, **kwargs
        Forwarded to :func:`matplotlib.pyplot.legend`.
    """
    h, l = plt.gca().get_legend_handles_labels()
    if not h:
        return None

    if not reverse:
        ret = plt.legend(*args, **kwargs)
    else:
        ret = plt.legend(h[::-1], l[::-1], *args, **kwargs)

    frame = ret.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_linewidth(lw)
    return ret


def get_palette(name=None, num_colors=8, start=0):
    """Get a color palette from matplotlib's colormap database

    Parameters
    ----------
    name : str, optional
        Name of the palette to get. If `None`, get the active palette.
    num_colors : int
        Number of colors to retrieve.
    start : int
        Staring from this color number.

    Returns
    -------
    List[color]
    """
    if not name:
        return [x["color"] for x in mpl.rcParams["axes.prop_cycle"]]

    brewer = dict(Set1=9, Set2=8, Set3=12, Pastel1=9, Pastel2=8, Accent=8, Dark2=8, Paired=12)
    if name in brewer:
        total = brewer[name]
        take = min(num_colors, total)
        bins = np.linspace(0, 1, total)[:take]
    else:
        bins = np.linspace(0, 1, num_colors + 2)[1:-1]

    cmap = plt.get_cmap(name)
    palette = cmap(bins)[:, :3]

    from itertools import cycle, islice
    palette = list(islice(cycle(palette), start, start + num_colors))
    return [list(color) for color in palette]


def set_palette(name=None, num_colors=8, start=0):
    """Set the active color palette

    Parameters
    ----------
    name : str, optional
        Name of the palette. If `None`, modify the active palette.
    num_colors : int
        Number of colors to retrieve.
    start : int
        Staring from this color number.
    """
    palette = get_palette(name, num_colors, start)
    mpl.rcParams["axes.prop_cycle"] = plt.cycler("color", palette)
    mpl.rcParams["patch.facecolor"] = palette[0]
    plt.gca().set_prop_cycle(mpl.rcParams["axes.prop_cycle"])


def direct_cmap_norm(data, colors, blend=1):
    """Colormap with direct mapping: data[i] -> colors[i]

    Parameters
    ----------
    data : array_like
        The data for which the colormap will be created.
    colors : color or tuple of colors
        Colors to map to unique data values.
    blend : float
        Like `alpha` but always blend with white.

    Returns
    -------
    Tuple[ListedColormap, BoundaryNorm]
    """
    if not isinstance(colors, (list, tuple)):
        colors = [colors]
    if blend < 1:
        colors = [blend_colors(c, "white", blend) for c in colors]

    # colormap with an boundary norm to match the unique data points
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)
    boundaries = np.append(np.unique(data), np.inf)
    norm = BoundaryNorm(boundaries, len(boundaries) - 1)

    return cmap, norm


def align(x, y):
    """Return text alignment based on (x, y) numbers

    Parameters
    ----------
    x, y : int
        Negative is left/bottom, positive is right/top, zero is center/center.

    Examples
    --------
    >>> align(1, -1)
    ('right', 'bottom')
    >>> align(0, 1)
    ('center', 'top')
    >>> align(-1, 0)
    ('left', 'center')
    """
    if np.isclose(x, 0):
        ha = "center"
    elif x > 0:
        ha = "right"
    else:
        ha = "left"

    if np.isclose(y, 0):
        va = "center"
    elif y > 0:
        va = "top"
    else:
        va = "bottom"

    return ha, va
