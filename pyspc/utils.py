import numpy as np


def is_iqr_outlier(x: np.ndarray, factor=1.5, **kwargs):
    """Find outliers based on inter-quartile range (IQR)"""
    x = np.array(x).reshape((-1,))
    q1, q3 = tuple(np.nanquantile(x, [0.25, 0.75], **kwargs))
    iqr = q3 - q1
    return (x < (q1 - factor * iqr)) | (x > (q3 + factor * iqr))
