from nose2.tests._common import TestCase

from pyCovariance.checking import check_positive, check_type, check_value


class TestChecking(TestCase):
    def test_check_positive(self):
        a = -5
        self.assertRaises(
            ValueError,
            lambda: check_positive(a, 'a', strictly=True)
        )

        a = -5
        self.assertRaises(
            ValueError,
            lambda: check_positive(a, 'a', strictly=False)
        )

        a = 0
        self.assertRaises(
            ValueError,
            lambda: check_positive(a, 'a', strictly=True)
        )

        a = 0
        check_positive(a, 'a', strictly=False)

        a = 5
        check_positive(a, 'a', strictly=True)

        a = 5
        check_positive(a, 'a', strictly=False)

    def test_check_type(self):
        a = -5
        self.assertRaises(
            TypeError,
            lambda: check_type(a, 'a', [float])
        )

        check_type(a, 'a', [int, float])

        a = [5, 2]
        self.assertRaises(
            TypeError,
            lambda: check_type(a, 'a', [int, float])
        )
        check_type(a, 'a', [list,  tuple])

    def test_check_value(self):
        a = -5
        self.assertRaises(
            ValueError,
            lambda: check_value(a, 'a', [2, 3])
        )
        check_value(a, 'a', [-5, 2])
