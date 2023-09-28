from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

__all__ = ["assert_spectraframe_equal"]


def assert_spectraframe_equal(sf1, sf2):
    assert_array_equal(sf1.spc, sf2.spc)
    assert_array_equal(sf1.wl, sf2.wl)
    assert_frame_equal(sf1.data, sf2.data)
