from pandas_munkres.PandasMunkres import PandasMunkres, MunkresElement
import pytest
import pandas as pd


class TestMunkres:
    def test_calculate(self, assignment_matrix):
        m = PandasMunkres(assignment_matrix)
        assert m.calculate() == [(0, 0), (1, 1), (2, 2)]


@pytest.fixture(scope="class")
def assignment_matrix():
    costs = {
        "jeff": [80, 40, 50, 46],
        "jill": [40, 70, 20, 25],
        "joan": [30, 10, 20, 30],
        "jonathan": [35, 20, 25, 30],
    }
    yield pd.DataFrame(data=costs)


class TestPandasMunkresAccessor:
    def test_cover(self):
        series = pd.Series([MunkresElement(1), MunkresElement(2), MunkresElement(3)])
        series.munkres.cover()
        assert series.apply(lambda x: x.covered).all()
        assert series.munkres.covered

    def test_uncover(self):
        series = pd.Series([MunkresElement(1), MunkresElement(2), MunkresElement(3)])
        series.munkres.cover()
        series.munkres.uncover()
        assert not series.apply(lambda x: x.covered).any()
        assert not series.munkres.covered
