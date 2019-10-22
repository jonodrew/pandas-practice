from typing import Tuple, List
import pandas as pd


@pd.api.extensions.register_series_accessor("munkres")
class PandasMunkresAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.covered = False

    def cover(self):
        """
        Marks all elements in this series as 'covered' and also itself
        :return:
        """
        self._obj.apply(lambda x: x.cover())
        self.covered = True

    def uncover(self):
        """
        Marks all elements in this series as uncovered and also itself
        :return:
        """
        self._obj.apply(lambda x: x.uncover())
        self.covered = False


class MunkresElement:
    def __init__(self, value: int):
        self.starred = False
        self.value = value
        self.covered = False
        self.times_covered = 0

    def cover(self):
        self.covered = True
        self.times_covered += 1

    def uncover(self):
        self.covered = False
        self.times_covered = 0

    def __str__(self):
        return f"Value: {self.value}"


class PandasMunkres:
    def __init__(self, matrix: pd.DataFrame):
        self.assignment_matrix = matrix
        self.assignment_matrix: pd.DataFrame = self.assignment_matrix.apply(self._convert_to_munkres_objects)
        self.lines_covered = 0

    def calculate(self) -> List[Tuple[int, int]]:
        self._step_1()
        self._step_2()
        print(self.assignment_matrix)
        self._step_3()
        return []

    @staticmethod
    def _convert_to_munkres_objects(series: pd.Series):
        return series.apply(func=lambda x: MunkresElement(value=x))

    def _step_1(self):
        """
        For each row in the matrix, find the smallest value. Then subtract that value from every value in the row
        :return:
        """
        self.assignment_matrix = self.assignment_matrix.apply(func=PandasMunkres._subtract_min_from_every_element, axis=1)

    def _step_2(self):
        """
        Repeat step 1 but with columns instead of rows
        :return:
        """
        self.assignment_matrix = self.assignment_matrix.apply(func=PandasMunkres._subtract_min_from_every_element, axis=0)

    def _step_3(self):
        """
        Draw lines through the row and columns that have the 0 entries such that the fewest lines possible are drawn.
        Calculate the number of zeroes in each row or column. Cover the series. Repeat, ignoring the covered series,
        until there are no more uncovered zeroes.
        :return:
        """
        while self.assignment_matrix.applymap(lambda x: x.value == 0 and not x.covered).any(axis=None):
            self._cover_series_with_most_zeroes()
        if self.lines_covered == len(self.assignment_matrix):
            print("Done")
            self._clear_covers()

    @staticmethod
    def _get_count_of_zeroes_from_series(series: pd.Series) -> int:
        return series.apply(lambda x: x.value if not x.covered else None).value_counts().get(0)

    @staticmethod
    def _subtract_min_from_every_element(series: pd.Series):
        min_value = series.copy().apply(func=lambda x: x.value).min()
        return series.apply(func=PandasMunkres._subtract_value_from_object_value, args=(min_value,))

    @staticmethod
    def _subtract_value_from_object_value(x: MunkresElement, value):
        x.value -= value
        return x

    def _cover_series_with_most_zeroes(self):
        rows_with_zeroes = self.assignment_matrix.apply(func=PandasMunkres._get_count_of_zeroes_from_series, axis=1)
        columns_with_zeroes = self.assignment_matrix.apply(func=PandasMunkres._get_count_of_zeroes_from_series, axis=0)
        most_zeroes_index = rows_with_zeroes.idxmax() \
            if rows_with_zeroes.max() > columns_with_zeroes.max() \
            else columns_with_zeroes.idxmax()
        self.assignment_matrix[most_zeroes_index].munkres.cover()
        self.lines_covered += 1

    def _count_covered_series(self):
        return self.assignment_matrix.apply(lambda x: x.munkres.covered, axis=0).all(axis=None) + self.assignment_matrix.apply(lambda x: x.munkres.covered, axis=1).all(axis=None)

    def _find_a_series_with_one_zero(self):
        for row_name, row in self.assignment_matrix.iterrows():
            if row.apply(lambda x: x.value == 0).count_values().get(0) == 1:
                return row_name
        for column_name, column in self.assignment_matrix.iteritems():
            if column.apply(lambda x: x.value == 0).count_values().get(0) == 1:
                return column_name

    def _clear_covers(self):
        self.assignment_matrix.apply(lambda x: x.munkres.uncover(), axis=1)
        self.assignment_matrix.apply(lambda x: x.munkres.uncover(), axis=0)

