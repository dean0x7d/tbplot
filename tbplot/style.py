from contextlib import suppress

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from .pltutils import get_palette
from .detail.utils import with_defaults

__all__ = ["use_style", "tbplot_style"]


def _make_style():
    nearly_black = "0.15"
    linewidth = 0.6
    dpi = 160
    palette = list(get_palette("Set1"))
    palette[5] = list(get_palette("Set2"))[5]

    # Use classic matplotlib (v1.x) as baseline
    defaults = mpl_style.library["classic"]

    # Modified properties -- the defaults are noted in brackets
    style = {
        "lines.solid_capstyle": "round",  # [projecting] butt|round|projecting
        "font.size": 7.0,  # [12.0]
        "text.color": nearly_black,  # [black]
        "mathtext.default": "regular",  # [it] the default font to use for math.
        "axes.edgecolor": nearly_black,  # [black] axes edge color
        "axes.linewidth": linewidth,  # [1.0] edge linewidth
        "axes.labelcolor": nearly_black,  # [black]
        "axes.unicode_minus": False,  # [True] use unicode for the minus symbol
        "axes.prop_cycle": plt.cycler("color", palette),  # ["bgrcmyk"]
        "patch.facecolor": palette[1],  # [b]
        "xtick.major.size": 2.5,  # [4] major tick size in points
        "xtick.minor.size": 1.0,  # [2] minor tick size in points
        "xtick.major.width": linewidth,  # [0.5] major tick width in points
        "xtick.color": nearly_black,  # [black] color of the tick labels
        "ytick.major.size": 2.5,  # [4] major tick size in points
        "ytick.minor.size": 1.0,  # [2] minor tick size in points
        "ytick.major.width": linewidth,  # [0.5] major tick width in points
        "ytick.color": nearly_black,  # [black] color of the tick labels
        "legend.fancybox": True,  # [False] Use a rounded box for the legend
        "legend.numpoints": 1,  # [2] the number of points in the legend line
        "legend.fontsize": "medium",  # ["large"]
        "legend.framealpha": 0.9,  # [None] opacity of of legend frame
        "figure.figsize": (3.4, 2.8),  # [(8, 6) inch] (3.4, 2.8) inch == (8.6, 7.1) cm
        "figure.dpi": dpi,  # [80] figure dots per inch
        "figure.facecolor": "white",  # [0.75] figure facecolor
        "image.cmap": "viridis",  # [jet...]
        "savefig.dpi": dpi,  # [100] figure dots per inch
        "savefig.bbox": "tight",  # ["standard"]
        "savefig.pad_inches": 0.04,  # [0.1] padding to be used when bbox is set to "tight"
    }

    return with_defaults(style, defaults)


tbplot_style = _make_style()


def _is_jupyter_notebook():
    """Detect if this is being executed inside of a notebook"""
    try:
        # noinspection PyUnresolvedReferences
        get_ipython()
        return True
    except NameError:
        return False


def _is_notebook_inline_backend():
    return _is_jupyter_notebook() and "backend_inline" in mpl.get_backend()


def _reset_notebook_inline_backend():
    with suppress(NameError):
        # noinspection PyUnresolvedReferences
        get_ipython().run_line_magic("matplotlib", "inline")


def use_style(style=tbplot_style):
    """use_style(style=tbplot_style)

    Shortcut for :func:`matplotlib.style.use` with tbplot style applied by default and
    special considerations for Jupyter notebook.

    Parameters
    ----------
    style : dict
        A matplotlib style specification.
    """
    mpl_style.use(style)

    # The style shouldn't override inline backend settings
    if _is_notebook_inline_backend():
        _reset_notebook_inline_backend()
