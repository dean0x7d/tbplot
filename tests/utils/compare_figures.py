import os
import shutil
import tempfile
from contextlib import suppress

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.units
from matplotlib.testing.compare import compare_images

from .path import path_from_fixture
from tbplot import tbplot_style


def _remove_text(figure):
    from matplotlib import ticker

    figure.suptitle("")
    for ax in figure.get_axes():
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        with suppress(AttributeError):
            ax.zaxis.set_major_formatter(ticker.NullFormatter())
            ax.zaxis.set_minor_formatter(ticker.NullFormatter())


class AssertFigure:
    """A context manager which compares a figure to a baseline image

    An `AssertionError` is raised if the images don't match up within tolerance.
    """

    def __init__(self, request):
        self.request = request
        self.passed = False

        self._original_rc = {}
        self._original_units_registry = {}

    def __call__(self, ext=".png", tol=10, remove_text=True, savefig_kwargs=None):
        self.ext = ext
        self.tol = tol
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs or {}
        return self

    def _enter_style(self, style=tbplot_style):
        self._original_rc = mpl.rcParams.copy()
        self._original_units_registry = matplotlib.units.registry.copy()

        matplotlib.style.use(style)
        mpl.use("Agg", warn=False)

    def _exit_style(self):
        mpl.rcParams.clear()
        mpl.rcParams.update(self._original_rc)
        matplotlib.units.registry.clear()
        matplotlib.units.registry.update(self._original_units_registry)

    def __enter__(self):
        self._enter_style()
        self.fig = plt.figure()
        return self

    def __exit__(self, exception, *_):
        if exception:
            return

        if self.remove_text:
            _remove_text(self.fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            actual_file = path_from_fixture(self.request, prefix=tmpdir, ext=self.ext)
            actual_filename = str(actual_file)
            if not actual_file.parent.exists():
                actual_file.parent.mkdir(parents=True)
            plt.savefig(actual_filename, **self.savefig_kwargs)

            baseline = path_from_fixture(self.request, prefix="baseline_plots", ext=self.ext)
            baseline_filename = str(baseline)

            if baseline.exists():
                try:
                    failure_data = compare_images(baseline_filename, actual_filename,
                                                  self.tol, in_decorator=True)
                except ValueError as exc:
                    if "could not be broadcast" not in str(exc):
                        raise
                    else:
                        failure_data = dict(actual=actual_filename, expected=baseline_filename)

                if failure_data:
                    self.report(failure_data)
                    raise AssertionError("Mismatch between actual figure and baseline image")
            else:
                os.makedirs(os.path.dirname(baseline_filename), exist_ok=True)
                shutil.copyfile(actual_filename, baseline_filename)
                self.passed = True
                self.report(None)

        plt.close()
        self._exit_style()

    def report(self, failure_data):
        def reportfile(variant):
            path = path_from_fixture(self.request, prefix="failed", variant=variant, ext=self.ext)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            return str(path)

        def delete(variant):
            filename = reportfile(variant)
            if os.path.exists(filename):
                with suppress(OSError):
                    os.remove(filename)

        if failure_data:
            shutil.copyfile(failure_data["actual"], reportfile("_actual"))
            shutil.copyfile(failure_data["expected"], reportfile("_baseline"))
            if "diff" in failure_data:
                shutil.copyfile(failure_data["diff"], reportfile("_diff"))
        else:
            delete("_actual")
            delete("_baseline")
            delete("_diff")
