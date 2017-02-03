import copy
import numpy as np


def with_defaults(options: dict, defaults_dict: dict = None, **defaults_kwargs):
    """Return a dict where missing keys are filled in by defaults

    >>> options = dict(hello=0)
    >>> with_defaults(options, hello=4, world=5) == dict(hello=0, world=5)
    True
    >>> defaults = dict(hello=4, world=5)
    >>> with_defaults(options, defaults) == dict(hello=0, world=5)
    True
    >>> with_defaults(options, defaults, world=7, yes=3) == dict(hello=0, world=5, yes=3)
    True
    """
    options = options if options else {}
    if defaults_dict:
        options = dict(defaults_dict, **options)
    return dict(defaults_kwargs, **options)


def x_pi(value):
    """Return str of value in 'multiples of pi' latex representation

    >>> x_pi(6.28) == r"$2\pi$"
    True
    >>> x_pi(3) == r"$0.95\pi$"
    True
    >>> x_pi(-np.pi) == r"$-\pi$"
    True
    >>> x_pi(0) == "0"
    True
    """
    n = value / np.pi
    if np.isclose(n, 0):
        return "0"
    elif np.isclose(abs(n), 1):
        return r"$\pi$" if n > 0 else r"$-\pi$"
    else:
        return r"${:.2g}\pi$".format(n)


class FuzzySet:
    """Like a regular `set`, but the items can be `np.ndarray` and the comparisons
    are approximate with a relative and absolute tolerance.
    """

    def __init__(self, iterable=None, rtol=1.e-3, atol=1.e-5):
        self.data = []
        self.rtol = rtol
        self.atol = atol

        if iterable:
            for item in iterable:
                self.add(item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(np.allclose(item, x, rtol=self.rtol, atol=self.atol) for x in self.data)

    def __iadd__(self, other):
        for item in other:
            self.add(item)
        return self

    def __add__(self, other):
        if isinstance(other, FuzzySet):
            ret = copy.copy(self)
            ret += other
            return ret
        else:
            return copy.copy(self)

    def __radd__(self, other):
        return self + other

    def add(self, item):
        if item not in self:
            self.data.append(item)
