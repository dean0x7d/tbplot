import pytest
from .utils.compare_figures import AssertFigure


@pytest.fixture
def assert_figure(request):
    """Compare a figure to a baseline image"""
    return AssertFigure(request)
