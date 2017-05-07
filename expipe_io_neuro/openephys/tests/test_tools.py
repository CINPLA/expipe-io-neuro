import pytest
import numpy as np
from expipe_io_neuro.openephys.tools import _start_from_zero_time, _zeros_to_nan, _cut_to_same_len

def test_start_from_zero_time():
    t = np.arange(11)
    t[0] = 10
    t[1] = 0.0
    x = np.arange(11)
    y = np.arange(11, 22)
    t_, (x_, y_) = _start_from_zero_time(t, x, y)
    assert np.array_equal(t_, t[1:])
    assert np.array_equal(x_, x[1:])
    assert np.array_equal(y_, y[1:])
