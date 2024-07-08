import numpy as np
import pandas as pd
import scipy
from pandas.testing import assert_frame_equal
import pytest

from pyspc.peaks import (
    _around_max_peak_fit,
    around_max_peak_fit,
    gaussian,
    gaussian_peak_fit,
    lorentzian,
    lorentzian_peak_fit,
    parabola_peak_fit,
)


def test_parabola_peak_fit():
    p = np.poly1d([-1, 4, -3])
    x = [1.7, 2.1, 2.4]
    y = p(x)

    fit = parabola_peak_fit(x, y)
    assert fit["x_max"] == pytest.approx(2)
    assert fit["y_max"] == pytest.approx(1)
    assert fit["a"] == pytest.approx(-1)
    assert fit["b"] == pytest.approx(4)
    assert fit["c"] == pytest.approx(-3)


def test_gaussian_peak_fit():
    # Prepare data
    x = np.linspace(1, 100, 50)
    y = gaussian(x, 100, 40, 50)
    y += 5 * np.exp(np.random.ranf(50))

    # Manual fitting
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    ref_fit, _ = scipy.optimize.curve_fit(gaussian, x, y, p0=[np.max(y), mean, sigma])

    fit = gaussian_peak_fit(x, y)

    assert fit["x_max"] == fit["x0"]
    assert fit["y_max"] == pytest.approx(fit["a"])
    assert fit["a"] == pytest.approx(ref_fit[0])
    assert fit["x0"] == pytest.approx(ref_fit[1])
    assert fit["sigma"] == pytest.approx(ref_fit[2])


def test_lorentzian_peak_fit():
    # Prepare data
    x = np.linspace(1, 100, 50)
    y = lorentzian(x, 100, 40, 50)
    y += 5 * np.exp(np.random.ranf(50))

    # Manual fitting
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    ref_fit, _ = scipy.optimize.curve_fit(lorentzian, x, y, p0=[np.max(y), mean, sigma])

    fit = lorentzian_peak_fit(x, y)

    assert fit["x_max"] == fit["x0"]
    assert fit["y_max"] == pytest.approx(fit["a"])
    assert fit["a"] == pytest.approx(ref_fit[0])
    assert fit["x0"] == pytest.approx(ref_fit[1])
    assert fit["gamma"] == pytest.approx(ref_fit[2])


def test__around_max_peak_fit():
    with pytest.raises(ValueError):
        _around_max_peak_fit(range(8), range(8), parabola_peak_fit, 2)

    with pytest.raises(ValueError):
        _around_max_peak_fit(range(8), range(8), parabola_peak_fit, 11)

    with pytest.raises(ValueError):
        _around_max_peak_fit(range(8), range(8), parabola_peak_fit, 4)

    y = np.array([2, 4, 4, 10, 5, 4, 3, 2])
    x = np.arange(9)
    window = np.arange(1, 6)
    fit = _around_max_peak_fit(x, y, parabola_peak_fit, 5)
    ref_fit = parabola_peak_fit(x[window], y[window])
    assert ref_fit == fit


def test_around_max_peak_fit():
    with pytest.raises(ValueError):
        around_max_peak_fit(range(8), range(8), "test", 2)

    y = np.array([2, 4, 4, 10, 5, 4, 3, 2])
    x = np.arange(9)
    window = np.arange(1, 6)

    # Max
    # res_default = around_max_peak_fit(x, y, "max")
    res_max = around_max_peak_fit(x, y, "max")
    # assert_frame_equal(res_default, res_max)
    assert_frame_equal(
        res_max,
        pd.DataFrame({"x_max": [3.0], "y_max": [10.0]}),
        check_dtype=False,
    )

    # Parabola
    res_parabola = around_max_peak_fit(x, y, "parabola", 5)
    ref_fit = parabola_peak_fit(x[window], y[window])
    assert_frame_equal(
        res_parabola, pd.DataFrame(ref_fit, index=[0]), check_dtype=False
    )

    # Gaussian
    res_gaussian = around_max_peak_fit(x, y, "gaussian", 5)
    ref_fit = gaussian_peak_fit(x[window], y[window])
    assert_frame_equal(
        res_gaussian, pd.DataFrame(ref_fit, index=[0]), check_dtype=False
    )

    # Lorentzian
    res_lorentzian = around_max_peak_fit(x, y, "lorentzian", 5)
    ref_fit = lorentzian_peak_fit(x[window], y[window])
    assert_frame_equal(
        res_lorentzian, pd.DataFrame(ref_fit, index=[0]), check_dtype=False
    )

    # Callable
    res_callable = around_max_peak_fit(x, y, lambda xx, yy: (np.max(yy),), 5)
    assert_frame_equal(
        res_callable, pd.DataFrame.from_records([(10,)]), check_dtype=False
    )

    res_callable = around_max_peak_fit(
        x, y, lambda xx, yy: {"y_median": np.median(yy)}, 5
    )
    assert_frame_equal(
        res_callable,
        pd.DataFrame({"y_median": [np.median(y[window])]}),
        check_dtype=False,
    )
