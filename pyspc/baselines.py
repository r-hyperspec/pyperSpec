import numpy as np
import scipy


def rubberband(x: np.array, y: np.array) -> np.ndarray:
    """Rubberband baseline correction

    Based on code from Stack Exchange (Author: Roman Kiselev)
    Source: https://dsp.stackexchange.com/a/30723/69444 (CC BY-SA 3.0)

    Parameters
    ----------
    x : np.array
        1D array of wavelengths
    y : np.array
        1D array of signal values

    Returns
    -------
    np.ndarray
        1D array of rubberband baseline. Has the same size as x and y
    """
    v = scipy.spatial.ConvexHull(np.array([x, y]).T).vertices
    v = np.roll(v, -v.argmin())
    v = v[: (v.argmax() + 1)]
    bl = np.interp(x, x[v], y[v])

    return bl
