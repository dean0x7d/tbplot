"""Processing and presentation of computed data

Result objects hold computed data and offer postprocessing and plotting functions
which are specifically adapted to the nature of the stored data.
"""
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from collections import namedtuple

from . import pltutils
from .detail.utils import with_defaults
from .structure import (structure_plot_properties, plot_hoppings, plot_sites,
                        plot_periodic_boundaries)

__all__ = ['SpatialMap', 'StructureMap']

Positions = namedtuple('Positions', 'x y z')
# noinspection PyUnresolvedReferences
Positions.__doc__ = """
Named tuple of arrays

Attributes
----------
x, y, z : array_like
    1D arrays of Cartesian coordinates
"""


class SpatialMap:
    """Represents some spatially dependent property: data mapped to site positions

    Attributes
    ----------
    data : array_like
        1D array of values which correspond to x, y, z coordinates.
    positions : Tuple[array_like, array_like, array_like]
        Lattice site positions. Named tuple with x, y, z fields, each a 1D array.
    sublattices : Optional[array_like]
        Sublattice ID for each position.
    """

    def __init__(self, data, positions, sublattices=None):
        self.data = np.atleast_1d(data)
        self.positions = Positions(*positions)  # maybe convert from tuple
        if sublattices is not None:
            self.sublattices = np.atleast_1d(sublattices)
        else:
            self.sublattices = np.zeros_like(self.data)

    @property
    def num_sites(self) -> int:
        """Total number of lattice sites"""
        return self.data.size

    @property
    def x(self) -> np.ndarray:
        """1D array of x coordinates"""
        return self.positions.x

    @property
    def y(self) -> np.ndarray:
        """1D array of y coordinates"""
        return self.positions.y

    @property
    def z(self) -> np.ndarray:
        """1D array of z coordinates"""
        return self.positions.z

    def __getitem__(self, idx):
        """Same rules as numpy indexing"""
        return self.__class__(self.data[idx], (v[idx] for v in self.positions),
                              self.sublattices[idx])

    def cropped(self, **limits):
        """Return a copy which retains only the sites within the given limits

        Parameters
        ----------
        **limits
            Attribute names and corresponding limits. See example.

        Returns
        -------
        StructureMap

        Examples
        --------
        Leave only the data where -10 <= x < 10 and 2 <= y < 4::

            new = original.cropped(x=[-10, 10], y=[2, 4])
        """
        idx = np.ones(self.num_sites, dtype=np.bool)
        for name, limit in limits.items():
            v = getattr(self, name)
            idx = np.logical_and(idx, v >= limit[0])
            idx = np.logical_and(idx, v < limit[1])

        return self[idx]

    def clipped(self, v_min, v_max):
        """Clip (limit) the values in the `data` array, see :func:`~numpy.clip`"""
        return self.__class__(np.clip(self.data, v_min, v_max), self.positions, self.sublattices)

    @staticmethod
    def _decorate_plot():
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        pltutils.despine(trim=True)

    def plot_pcolor(self, **kwargs):
        """Color plot of the xy plane

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tripcolor`.
        """
        x, y, _ = self.positions
        kwargs = with_defaults(kwargs, shading="gouraud", rasterized=True)
        pcolor = plt.tripcolor(x, y, self.data, **kwargs)
        self._decorate_plot()
        return pcolor

    def plot_contourf(self, num_levels=50, **kwargs):
        """Filled contour plot of the xy plane

        Parameters
        ----------
        num_levels : int
            Number of contour levels.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tricontourf`.
        """
        levels = np.linspace(self.data.min(), self.data.max(), num=num_levels)
        x, y, _ = self.positions
        kwargs = with_defaults(kwargs, levels=levels, rasterized=True)
        contourf = plt.tricontourf(x, y, self.data, **kwargs)
        self._decorate_plot()
        return contourf

    def plot_contour(self, **kwargs):
        """Contour plot of the xy plane

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.tricontour`.
        """
        x, y, _ = self.positions
        contour = plt.tricontour(x, y, self.data, **kwargs)
        self._decorate_plot()
        return contour


class StructureMap(SpatialMap):
    """A subclass of :class:`.SpatialMap` that also includes hoppings between sites

    Attributes
    ----------
    hoppings : :class:`~scipy.sparse.csr_matrix`
        Sparse matrix of hopping IDs. See :attr:`.System.hoppings`.
    boundaries : List[:class:`~scipy.sparse.csr_matrix`]
        Boundary hoppings. See :attr:`.System.boundaries`.
    """

    def __init__(self, data, positions, sublattices, hoppings, boundaries=()):
        super().__init__(data, positions, sublattices)
        self.hoppings = hoppings
        self.boundaries = boundaries

    @classmethod
    def from_system(cls, data, system):
        return cls(data, system.positions, system.sublattices,
                   system.hoppings.tocsr(), system.boundaries)

    @property
    def spatial_map(self) -> SpatialMap:
        """Just the :class:`SpatialMap` subset without hoppings"""
        return SpatialMap(self.data, self.positions, self.sublattices)

    @staticmethod
    def _filter_csr_matrix(csr, idx):
        """Indexing must preserve all data, even zeros"""
        m = copy(csr)  # shallow copy
        m.data = m.data.copy()
        m.data += 1  # increment by 1 to preserve zeroes when slicing
        m = m[idx][:, idx]
        m.data -= 1
        return m

    @staticmethod
    def _filter_boundary(boundary, idx):
        b = copy(boundary)
        b.hoppings = StructureMap._filter_csr_matrix(b.hoppings, idx)
        return b

    def __getitem__(self, idx):
        """Same rules as numpy indexing"""
        return self.__class__(self.data[idx], (v[idx] for v in self.positions),
                              self.sublattices[idx], self._filter_csr_matrix(self.hoppings, idx),
                              [self._filter_boundary(b, idx) for b in self.boundaries])

    def plot(self, cmap="YlGnBu", site_radius=(0.03, 0.05), num_periods=1, **kwargs):
        """Plot the spatial structure with a colormap of :attr:`data` at the lattice sites

        Both the site size and color are used to display the data.

        Parameters
        ----------
        cmap : str
            Matplotlib colormap to be used for the data.
        site_radius : Tuple[float, float]
            Min and max radius of lattice sites. This range will be used to visually
            represent the magnitude of the data.
        num_periods : int
            Number of times to repeat periodic boundaries.
        **kwargs
            Additional plot arguments as specified in :func:`.structure_plot_properties`.
        """
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def to_radii(data):
            if not isinstance(site_radius, (tuple, list)):
                return site_radius

            positive_data = data - data.min()
            maximum = positive_data.max()
            if not np.allclose(maximum, 0):
                delta = site_radius[1] - site_radius[0]
                return site_radius[0] + delta * positive_data / maximum
            else:
                return site_radius[1]

        props = structure_plot_properties(**kwargs)
        props["site"] = with_defaults(props["site"], radius=to_radii(self.data), cmap=cmap)
        collection = plot_sites(self.positions, self.data, **props["site"])

        hop = self.hoppings.tocoo()
        props["hopping"] = with_defaults(props["hopping"], color="#bbbbbb")
        plot_hoppings(self.positions, hop, **props["hopping"])

        props["site"]["alpha"] = props["hopping"]["alpha"] = 0.5
        plot_periodic_boundaries(self.positions, hop, self.boundaries, self.data,
                                 num_periods, **props)

        pltutils.despine(trim=True)
        pltutils.add_margin()

        if collection:
            plt.sci(collection)
        return collection
