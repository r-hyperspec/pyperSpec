from typing import Any, Optional, Union, Tuple

import numpy as np
import pandas as pd

__all__ = ["SpectraFrame"]


class SpectraFrame:
    # ----------------------------------------------------------------------
    # Constructor

    def __init__(  # noqa: C901
        self,
        spc: np.ndarray,
        wl: Optional[np.ndarray] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        # Prepare SPC
        spc = np.array(spc)
        if spc.ndim == 1:
            spc = spc.reshape(1, -1)
        elif spc.ndim > 2:
            raise ValueError("Invalid spc is provided!")

        # Prepare wl
        if wl is None:
            wl = np.arange(spc.shape[1])
        else:
            wl = np.array(wl)
            if wl.ndim > 1:
                raise ValueError("Invalid wl is provided")

        # Parse data
        if data is None:
            data = pd.DataFrame(index=range(len(spc)), columns=None)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Checks
        if spc.shape[1] != len(wl):
            raise ValueError(
                "length of wavelength must be equal to number of columns in spc"
            )

        if spc.shape[0] != data.shape[0]:
            raise ValueError(
                "data must have the same number of instances(rows) as spc has"
            )

        self.spc = spc
        self.wl = wl
        self.data = data

    # ----------------------------------------------------------------------
    # Internal helpers

    def _parse_string_or_column_param(
        self, param: Union[str, pd.Series, np.ndarray, list, tuple]
    ) -> pd.Series:
        if isinstance(param, str) and (param in self.data.columns):
            return self.data[param]
        elif isinstance(param, pd.Series) and (param.shape[0] == self.nspc):
            return param
        elif (
            isinstance(param, np.ndarray)
            and (param.ndim == 1)
            and (param.shape[0] == self.nspc)
        ):
            return pd.Series(param)
        elif isinstance(param, (list, tuple)) and (len(param) == self.nspc):
            return pd.Series(param)
        else:
            raise TypeError(
                "Invalid parameter. It must be either a string of a data "
                "column name or pd.Series / np.array / list / tuple of "
                "lenght equal to number of spectra. "
            )

    # ----------------------------------------------------------------------
    # Properties for a quick access

    @property
    def shape(self) -> Tuple[int, int, int]:
        """A tuple representing the dimensionality of the Spectra

        Returns
        -------
        Tuple[int, int, int]:
            A tuple of the following structure:
            1. number of spectra (i.e. number of rows)
            2. number of data columns
            3. number of wavelength points
        """
        return self.nspc, self.data.shape[1], self.nwl

    @property
    def nwl(self) -> int:
        """Number of wavelength points"""
        return len(self.wl)

    @property
    def nspc(self) -> int:
        """Number of spectra in the object"""
        return self.spc.shape[0]

    @property
    def is_equally_spaced(self) -> bool:
        """Are wavelength values equally spaced?"""
        return len(np.unique(self.wl[1:] - self.wl[:-1])) == 1

    # ----------------------------------------------------------------------
    # Coping

    def copy(self) -> "SpectraFrame":
        return SpectraFrame(
            spc=self.spc.copy(), wl=self.wl.copy(), data=self.data.copy()
        )

    # ----------------------------------------------------------------------
    # Get and set items
    def _parse_getitem_selector(
        self, index: pd.Index, selector: Any
    ) -> Union[slice, list]:
        if isinstance(selector, slice):
            return index.slice_indexer(selector.start, selector.stop, selector.step)

        idx = index.get_indexer(selector)
        if np.any(idx == -1):
            raise ValueError(f"Unexpected wavelenght(s) {index[idx == -1]}")
        return idx

    def _parse_slicer(self, slicer: tuple) -> tuple:
        if not ((type(slicer) == tuple) and (len(slicer) in [3, 4])):
            raise ValueError(
                "Invalid subset value. Provide 3 values in format <row, column, wl>"
                "or 4 values in format <row, column, wl, True/False>"
            )

        if len(slicer) == 3:
            use_iloc = False
        else:
            use_iloc == bool(slicer[3])
            slicer = slicer[:3]

        rows, cols, wls = (
            [x]
            if (np.size(x) == 1) and (not isinstance(x, (slice, list, tuple)))
            else x
            for x in slicer
        )

        # From wl to indices
        if use_iloc:
            row_selector = rows
            col_selector = cols
            wl_selector = wls
        else:
            row_selector = self._parse_getitem_selector(self.data.index, rows)
            col_selector = self._parse_getitem_selector(self.data.columns, cols)
            wl_selector = self._parse_getitem_selector(pd.Index(self.wl), wls)

        return row_selector, col_selector, wl_selector

    def __setitem__(self, given: tuple[Any, Any, Any], value: Any) -> None:
        # row_slice, col_slice, wl_slice = self._parse_slicer(given)

        # if _is_empty_slice(col_slice) and not _is_empty_slice(wl_slice):
        #     self.spc[row_slice, wl_slice] = value
        # elif not _is_empty_slice(col_slice) and _is_empty_slice(wl_slice):
        #     self.data[row_slice, col_slice] = value
        # else:
        #     raise ValueError("Either data columns or wavelengths indexes must be `:`")
        raise NotImplementedError(
            "Not implemented. Please use `.data` or `.spc` directly."
        )

    def __getitem__(self, given: tuple) -> "SpectraFrame":
        row_slice, col_slice, wl_slice = self._parse_slicer(given)
        return SpectraFrame(
            spc=self.spc[row_slice, wl_slice],
            wl=self.wl[wl_slice],
            data=self.data.iloc[row_slice, col_slice],
        )

    # ----------------------------------------------------------------------
    # Arithmetic operations +, -, *, /, **, abs, round, ceil, etc.

    def __add__(self, other: Any) -> "SpectraFrame":
        if isinstance(other, type(self)):
            other = other.spc
        return SpectraFrame(spc=self.spc.__add__(other), wl=self.wl, data=self.data)

    def __sub__(self, other: Any) -> "SpectraFrame":
        if isinstance(other, type(self)):
            other = other.spc
        return SpectraFrame(spc=self.spc.__sub__(other), wl=self.wl, data=self.data)

    def __mul__(self, other: Any) -> "SpectraFrame":
        if isinstance(other, type(self)):
            other = other.spc
        return SpectraFrame(spc=self.spc.__mul__(other), wl=self.wl, data=self.data)

    def __truediv__(self, other: Any) -> "SpectraFrame":
        if isinstance(other, type(self)):
            other = other.spc
        return SpectraFrame(spc=self.spc.__truediv__(other), wl=self.wl, data=self.data)

    def __pow__(self, other: Any) -> "SpectraFrame":
        return SpectraFrame(spc=self.spc.__pow__(other), wl=self.wl, data=self.data)

    def __radd__(self, other: Any) -> "SpectraFrame":
        return SpectraFrame(spc=self.spc.__radd__(other), wl=self.wl, data=self.data)

    def __rsub__(self, other: Any) -> "SpectraFrame":
        return SpectraFrame(spc=self.spc.__rsub__(other), wl=self.wl, data=self.data)

    def __rmul__(self, other: Any) -> "SpectraFrame":
        return SpectraFrame(spc=self.spc.__rmul__(other), wl=self.wl, data=self.data)

    def __rtruediv__(self, other: Any) -> "SpectraFrame":
        return SpectraFrame(
            spc=self.spc.__rtruediv__(other), wl=self.wl, data=self.data
        )

    # def __iadd__(self, other: Any) -> None:
    #     if isinstance(other, type(self)):
    #         other = other.spc
    #     self.spc = self.spc.__add__(other)

    # def __isub__(self, other: Any) -> None:
    #     if isinstance(other, type(self)):
    #         other = other.spc
    #     self.spc = self.spc.__sub__(other)

    # def __imul__(self, other: Any) -> None:
    #     if isinstance(other, type(self)):
    #         other = other.spc
    #     self.spc = self.spc.__mul__(other)

    # def __itruediv__(self, other: Any) -> None:
    #     if isinstance(other, type(self)):
    #         other = other.spc
    #     self.spc = self.spc.__truediv__(other)

    def __abs__(self) -> "SpectraFrame":
        return SpectraFrame(spc=np.abs(self.spc), wl=self.wl, data=self.data)

    def __round__(self, n: int) -> "SpectraFrame":
        return SpectraFrame(spc=np.round(self.spc, n), wl=self.wl, data=self.data)

    def __floor__(self) -> "SpectraFrame":
        return SpectraFrame(spc=np.floor(self.spc), wl=self.wl, data=self.data)

    def __ceil__(self) -> "SpectraFrame":
        return SpectraFrame(spc=np.ceil(self.spc), wl=self.wl, data=self.data)

    def __trunc__(self) -> "SpectraFrame":
        return SpectraFrame(spc=np.trunc(self.spc), wl=self.wl, data=self.data)

    # ----------------------------------------------------------------------

    def __str__(self) -> str:
        return str(self.shape)
