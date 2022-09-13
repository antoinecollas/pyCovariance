from numpy import testing as np_testing

RTOL = 1e-7
ATOL = 1e-10


def assert_allclose(actual, desired, rtol=RTOL, atol=ATOL, *args, **kwargs):
    return np_testing.assert_allclose(
        actual, desired, rtol, atol, *args, **kwargs)
