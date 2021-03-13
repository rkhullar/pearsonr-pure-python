import statistics
from unittest import TestCase

from parameterized import parameterized

from ..main import pearsonr


class MathTest(TestCase):

    @parameterized.expand([
        ([1, 2, 3, 4, 5], 3),
        ([92, 84, 64, 93, 55, 58, 4, 9, 17, 57, 88, 1, 85], 58)
    ])
    def test_median(self, arr, expected):
        actual = statistics.median(arr)
        self.assertEqual(actual, expected)

    @parameterized.expand([
        ([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4], -0.7426106572325057, 0.15055580885344547),
        ([1, 18, 50, 52, 60, 70, 20, 63], [8, 5, 27, 19, 83, 1, 90, 46], 0.08679698052295078, 0.8380711938799545),
        ([6, 73, 53, 85, 44, 50, 49], [97, 67, 33, 11, 5, 1, 20], -0.4904111369649974, 0.2638441794219994),
        ([45, 27, 98, 41, 54, 28, 22, 39], [88, 45, 10, 19, 68, 3, 38, 87], -0.16295811396891266, 0.6998197044230758),
        ([11, 95, 73, 81, 78, 88, 4], [77, 56, 73, 68, 64, 96, 73], -0.14490958566600073, 0.7565686061778427),
        ([44, 69, 95, 80, 86], [16, 4, 82, 61, 17], 0.6236560180120905, 0.2609339313668575),
        ([97, 56, 12, 91, 85, 41], [13, 26, 1, 85, 23, 25], 0.50080763818522, 0.311591896514609),
        ([12, 91, 57, 46, 84], [31, 94, 34, 89, 83], 0.6938543704389364, 0.19373190490472003),
        ([27, 77, 46, 95, 80, 21, 17], [57, 37, 2, 14, 53, 18, 23], 0.061045713792577165, 0.8965585705015693),
        ([87, 70, 90, 52, 64, 9, 62], [13, 26, 74, 92, 41, 81, 17], -0.4763662480735743, 0.279839235169021)
    ])
    def test_pearsonr_rand(self, x_arr, y_arr, expected_r, expected_p):
        actual_r, actual_p = pearsonr(x_arr, y_arr)
        self.assertAlmostEqual(actual_r, expected_r)
        self.assertAlmostEqual(actual_p, expected_p)

    @parameterized.expand([
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0, 0.0),
        ([25, 44, 22, 40], [25, 44, 22, 40], 1.0, 0.0),
        ([-2, 2, -2, 2], [2, -2, -2, 2], 0.0, 1.0),
        ([-2, 2, -4, 4], [2, -2, -4, 4], 0.6, 0.4)
    ])
    def test_pearsonr_spec(self, x_arr, y_arr, expected_r, expected_p):
        actual_r, actual_p = pearsonr(x_arr, y_arr)
        self.assertAlmostEqual(actual_r, expected_r)
        self.assertAlmostEqual(actual_p, expected_p)

    def test_pearson_none(self):
        x_arr, y_arr = [False], [0]
        p, r = pearsonr(x_arr, y_arr)
        self.assertIsNone(p)
        self.assertIsNone(r)
