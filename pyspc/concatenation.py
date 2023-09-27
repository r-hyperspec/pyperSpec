import numpy as np
import pandas as pd

from .spectra import SpectraFrame


def concat(*objs: SpectraFrame, axis=None) -> SpectraFrame:
    # TODO: Allow concatenating np.ndarray and pd.DataFrame
    is_same_wls = len(set(tuple(sf.wl.astype(np.float64)) for sf in objs)) == 1
    is_same_nspc = len(set(sf.nspc for sf in objs)) == 1

    if axis is None:
        # If all objects have the same number of rows and different number of
        # wavelengths then assume that we want to stack by columns
        if (not is_same_wls) and is_same_nspc:
            axis = 1
        else:
            axis = 0

    if axis == 0:
        if is_same_wls:
            return SpectraFrame(
                np.concatenate([sf.spc for sf in objs], axis=0),
                wl=objs[0].wl,
                data=pd.concat([sf.data for sf in objs], axis=0, ignore_index=True),
            )
        else:
            raise ValueError("Spectral data has different wavelenghts")
    elif axis == 1:
        # TODO: Order of wl
        # TODO: Check that wl ranges do not overlap
        return SpectraFrame(
            np.concatenate([sf.spc for sf in objs], axis=1),
            wl=np.concatenate([sf.wl for sf in objs], axis=None),
            data=pd.concat([sf.data for sf in objs], axis=1, ignore_index=True),
        )
    else:
        raise ValueError(f"Unexpected `axis` {axis}.")
