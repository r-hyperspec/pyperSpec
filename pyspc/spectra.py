from typing import Any, Optional, Union, Tuple, Callable

from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import scipy
import pybaselines
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D

from .baselines import rubberband

__all__ = ["SpectraFrame"]


def _is_empty_slice(param: Any) -> bool:
    """Is `param` an empty slice"""
    return (
        isinstance(param, slice)
        and (param.start is None)
        and (param.stop is None)
        and (param.step is None)
    )


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
    # Accessing data
    def _parse_getitem_single_selector(
        self, index: pd.Index, selector: Any, iloc: bool = False
    ) -> Union[slice, list]:
        # If selector is slicer like 400:600 or 0:8:2
        if isinstance(selector, slice):
            if _is_empty_slice(selector) or iloc:
                return selector
            else:
                return index.slice_indexer(selector.start, selector.stop, selector.step)

        # Pre-format selector to be 1d np.array
        selector: np.ndarray = np.array(selector)
        if selector.ndim == 0:
            selector = selector.reshape((-1,))

        # If boolean vector, e.g. [True, False, True]
        if selector.dtype == "bool":
            if len(selector) != len(index):
                raise ValueError("Boolean vector has does not match with the size")
            return np.where(selector)[0]

        # List of specific items to select, e.g. ["A", "D"], [0,-1]
        if iloc:
            idx = selector
        else:
            idx = index.get_indexer(selector)
            if np.any(idx == -1):
                raise ValueError(f"Unexpected selector {index[idx == -1]}")

        return idx

    def _parse_getitem_tuple(self, slicer: tuple) -> tuple:
        if not ((type(slicer) == tuple) and (len(slicer) in [3, 4])):
            raise ValueError(
                "Invalid subset value. Provide 3 values in format <row, column, wl>"
                "or 4 values in format <row, column, wl, True/False>"
            )

        use_iloc = False
        if len(slicer) == 4:
            use_iloc = bool(slicer[3])
            slicer = slicer[:3]

        rows, cols, wls = slicer

        # From labels to indices
        row_selector = self._parse_getitem_single_selector(
            self.data.index, rows, iloc=use_iloc
        )
        col_selector = self._parse_getitem_single_selector(
            self.data.columns, cols, iloc=use_iloc
        )
        wl_selector = self._parse_getitem_single_selector(
            pd.Index(self.wl), wls, iloc=use_iloc
        )

        return row_selector, col_selector, wl_selector

    def __setitem__(self, given: Union[str, tuple], value: Any) -> None:
        if isinstance(given, str):
            return self.data.__setitem__(given, value)

        row_slice, col_slice, wl_slice = self._parse_getitem_tuple(given)
        if _is_empty_slice(col_slice) and not _is_empty_slice(wl_slice):
            self.spc[row_slice, wl_slice] = value
        elif not _is_empty_slice(col_slice) and _is_empty_slice(wl_slice):
            self.data.iloc[row_slice, col_slice] = value
        else:
            raise ValueError(
                "Invalid slicing. Either data columns or "
                "wavelengths indexes must be `:`"
            )

    def __getitem__(self, given: Union[str, tuple]) -> "SpectraFrame":
        if isinstance(given, str):
            return self.data[given]

        row_slice, col_slice, wl_slice = self._parse_getitem_tuple(given)
        return SpectraFrame(
            spc=self.spc[row_slice, wl_slice],
            wl=self.wl[wl_slice],
            data=self.data.iloc[row_slice, col_slice],
        )

    def __getattr__(self, name) -> pd.Series:
        return getattr(self.data, name)

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
    # Wavelengths

    def resample_wl(
        self, new_wl: np.ndarray, method="interp1d", **kwargs
    ) -> "SpectraFrame":
        """Resample wavelengths, i.e. shift wavelenghts with interpolation

        Parameters
        ----------
        new_wl : np.ndarray
            New wavenumbers
        method : str, optional
            Method for interpolation. Currently only "interp1d" is supported.
            Which is using `scipy.interpolate.interp1d` function.
        kwargs : dict, optional
            Additional parameters to be passed to the interpolator function.

        Returns
        -------
        SpectraFrame
            A new SpectraFrame object with `new_wl` as wavenumbers, and
            interpolated signal values as spectral data. `*.data` part
            remains the same.

        Raises
        ------
        NotImplementedError
            Unimplemented method of interpolation.
        """
        if method == "interp1d":
            interpolator = scipy.interpolate.interp1d(x=self.wl, y=self.spc, **kwargs)
            new_spc = interpolator(new_wl)
        else:
            raise NotImplementedError("Other methods not available yet")

        return SpectraFrame(new_spc, wl=new_wl, data=self.data)

    # ----------------------------------------------------------------------
    # Stats & Applys
    def _get_axis(self, axis, groupby=None) -> int:
        """Get axis value in standard format"""
        if groupby is not None:
            return 0
        if axis in [0, "index"]:
            return 0
        elif axis in [1, "columns"]:
            return 1
        else:
            raise ValueError(f"Unexpected `axis` value {axis}")

    def _get_groupby(self, groupby) -> list[str]:
        """Format and validate groupby value"""
        if groupby is None:
            return None

        # Grouped
        if isinstance(groupby, str):
            groupby = [groupby]

        # Check the names are in the data
        for name in groupby:
            if name not in self.data.columns:
                raise ValueError(f"Column '{name}' is not presented in the data")

        return groupby

    def _apply_func(
        self,
        func: Union[str, Callable],
        *args,
        data: np.ndarray = None,
        axis: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Apply a function alog an axis

        Dispatches calculation to `np.apply_alog_axis` (if func is callable) or
        `np.<func>` (if func is a string)

        Parameters
        ----------
        func : Union[str, Callable]
            Either a string with the name of numpy funciton, e.g "max", "mean", etc.
            Or a callable function that can be passed to `numpy.apply_along_axis`
        data : np.ndarray, optional
            To which data apply the function, by default `self.spc`
            This parameter is useful for cases when the function must be applied on
            different parts of the spctral data, e.g. when groupby is used
        axis : int, optional
            Standard axis. Same as in `numpy` or `pandas`, by default 1

        Returns
        -------
        np.ndarray
            The output array. The shape of out is identical to the shape of data, except
            along the axis dimension. This axis is removed, and replaced with new
            dimensions equal to the shape of the return value of func. So if func
            returns a scalar, the output will be eirther single row (axis=0) or
            single column (axis=1) matrix.

        Raises
        ------
        ValueError
            Function with provided name `func` was not found in `numpy`
        """
        # Check and prepare parameters
        if data is None:
            data = self.spc

        if isinstance(func, str):
            name = func
            if hasattr(np, name):
                func = getattr(np, name)
            else:
                raise ValueError(f"Could not find function {name} in `numpy`")

            res: np.ndarray = func(data, *args, axis=axis, **kwargs)
            # Functions like np.quantile behave differently than apply_alog_axis
            # Here we make the shape of the matrix to be the same
            if (res.ndim > 1) and (axis == 1):
                res = res.T
        else:
            res = np.apply_along_axis(func, axis, data, *args, **kwargs)

        # Reshape the result to keep dimenstions
        if res.ndim == 1:
            res = res.reshape((1, -1)) if axis == 0 else res.reshape((-1, 1))

        return res

    def apply(
        self,
        func: Union[str, callable],
        *args,
        groupby: Union[str, list[str]] = None,
        axis: int = 0,
        **kwargs,
    ) -> "SpectraFrame":
        """Apply function to the spectral data

        Parameters
        ----------
        func : Union[str, callable]
            Either a string with the name of numpy funciton, e.g "max", "mean", etc.
            Or a callable function that can be passed to `numpy.apply_along_axis`
        groupby : Union[str, list[str]], optional
            Single or list of `data` column names to use for grouping the data.
            By default None, so the function applied to the all spectral data.
        axis : int, optional
             Standard axis. Same as in `numpy` or `pandas`, by default 1 when groupby
             is not provided, and 0 when provided.

        Returns
        -------
        SpectraFrame
            Output spectral frame where
            * `out.spc` is the results of `func`
            * `out.wl` either the same (axis=0 OR axis=1 and `nwl` matches)
              or range 0..N (axis=1 and `nwl` does not match)
            * `out.data` The same if axis=1. If axis=0, either empty (no grouping)
                or represents the grouping.
        """

        # Prepare arguments
        axis = self._get_axis(axis, groupby)
        groupby = self._get_groupby(groupby)

        # Prepare default values
        new_wl = self.wl if axis == 0 else None
        new_data = self.data if axis == 1 else None

        if groupby is None:
            new_spc = self._apply_func(func, *args, axis=axis, **kwargs)
        else:
            # Prepare a dataframe for groupby aggregation
            grouped = self.to_pandas().groupby(groupby)[self.wl]

            # Prepare list of group names as dicts {'column name': 'column value', ...}
            if len(groupby) > 1:
                groups = [dict(zip(groupby, gr)) for gr in grouped.groups.keys()]
            else:
                groups = [{groupby[0]: gr} for gr in grouped.groups.keys()]

            # Apply to each group
            spc_list = [
                self._apply_func(func, *args, data=group.values, axis=0, **kwargs)
                for _, group in grouped
            ]
            data_list = [
                pd.DataFrame({**gr, "group_index": range(spc_list[i].shape[0])})
                for i, gr in enumerate(groups)
            ]

            # Combine
            new_spc = np.concatenate(spc_list, axis=0)
            new_data = pd.concat(data_list, axis=0, ignore_index=True)

        # If the applied function returns same number of wavelenghts
        # we assume that wavelengths are the same, e.g. baseline,
        # smoothing, etc.
        if (new_wl is None) and (new_spc.shape[1] == self.nwl):
            new_wl = self.wl

        return SpectraFrame(new_spc, wl=new_wl, data=new_data)

    def area(self) -> "SpectraFrame":
        """Calculate area under the spectra"""
        return SpectraFrame(
            scipy.integrate.trapezoid(self.spc, x=self.wl, axis=1).reshape((-1, 1)),
            wl=None,
            data=self.data,
        )

    # ----------------------------------------------------------------------
    # Dispatching to numpy methods
    # TODO: It would be good to group the method declarations below

    def min(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "min" if not ignore_na else "nanmin"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def max(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "max" if not ignore_na else "nanmax"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def sum(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "sum" if not ignore_na else "nansum"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def mean(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "mean" if not ignore_na else "nanmean"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def std(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "std" if not ignore_na else "nanstd"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def median(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "median" if not ignore_na else "nanmedian"
        return self.apply(func, *args, groupby=groupby, axis=axis, **kwargs)

    def mad(
        self, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        if ignore_na:
            median = lambda x: np.nanmedian(x, *args, **kwargs)
        else:
            median = lambda x: np.median(x, *args, **kwargs)
        return self.apply(
            lambda x: median(np.absolute(x - median(x))), groupby=groupby, axis=axis
        )

    def quantile(
        self, q, *args, groupby=None, axis=1, ignore_na=False, **kwargs
    ) -> "SpectraFrame":
        func = "quantile" if not ignore_na else "nanquantile"
        return self.apply(func, q, *args, groupby=groupby, axis=axis, **kwargs)

    # ----------------------------------------------------------------------
    # Manipulations
    def normalize(self, method: str, ignore_na: bool = True) -> "SpectraFrame":
        """Dispatcher for spectra normalization

        Parameters
        ----------
        method : str
            Method of normaliztion. Available options: '01', 'area', 'vector', 'mean'
        ignore_na : bool, optional
            Ignore NaN values in the data, by default True

        Returns
        -------
        SpectraFrame
            A new SpectraFrame with normalized values

        Raises
        ------
        NotImplementedError
            Unknown or not implemented methods, e.g. peak normalization
        """
        spc = self.copy()
        if method == "01":
            spc = spc - spc.min(axis=1, ignore_na=ignore_na)
            spc = spc / spc.max(axis=1, ignore_na=ignore_na)
        elif method == "area":
            spc = spc / spc.area()
        elif method == "peak":
            raise NotImplementedError("Method not implemented yet")
        elif method == "vector":
            if ignore_na:
                spc = spc / np.sqrt(
                    np.nansum(np.power(spc.spc, 2), axis=1, keepdims=True)
                )
            else:
                spc = spc / np.sqrt(np.sum(np.power(spc.spc, 2), axis=1, keepdims=True))
        elif method == "mean":
            spc = spc / spc.mean(axis=1, ignore_na=ignore_na)

        return spc

    def smooth(self, method: str = "savgol", **kwargs) -> "SpectraFrame":
        """Dispatcher for spectra smoothing

        Parameters
        ----------
        method : str, optional
            Method of smoothing. Currently, only "savgol" is avalialbe
        kwargs : dict
            Additional parameters to pass to the smoothing method

        Returns
        -------
        SpectraFrame
            A new frame with smoothed values

        Raises
        ------
        NotImplementedError
            Unknown or unimplemented smoothing method
        """
        spc = self.copy()
        if method == "savgol":
            spc.spc = scipy.signal.savgol_filter(spc.spc, **kwargs)
        else:
            raise NotImplementedError("Method is not implemented yet")

        return spc

    def baseline(self, method: str, **kwargs) -> "SpectraFrame":
        """Dispatcher for spectra baseline estimation

        Dispatches baseline correction to the corresponding method
        in `pybaselines` package.
        In addition, "rubberband" method is available.

        Parameters
        ----------
        method : str
            A name of the method in `pybaselines` package (e.g. "airpls", "snip"),
            or "rubberband"
        kwargs: dict
            Additional parameters to pass to the baseline correction method

        Returns
        -------
        SpectraFrame
            A frame of estimated baselines

        Raises
        ------
        ValueError
            Unknown baseline method provided
        """
        baseline_fitter = pybaselines.Baseline(x_data=self.wl)
        if hasattr(baseline_fitter, method):
            baseline_method = getattr(baseline_fitter, method)
            baseline_func = lambda y: baseline_method(y, **kwargs)[0]
        elif method == "rubberband":
            baseline_func = lambda y: rubberband(self.wl, y, **kwargs)
        else:
            raise ValueError(
                "Unknown method. Method must be either "
                "from `pybaselines` or 'rubberband'"
            )
        return self.apply(baseline_func, axis=1)

    # ----------------------------------------------------------------------
    # Format conversion

    def to_pandas(self, multiindex=False, string_names=False) -> pd.DataFrame:
        """Convert to a pandas DataFrame

        Parameters
        ----------
        multiindex : bool, optional
            Adds an index level to columns separating spectral data (`spc`) from
            meta data (`data`), by default False

        Returns
        -------
        pd.DataFrame
            Dataframe where spectral data is combined with meta data.
            Wavelengths are used as column names for spectral data part.
        """
        df = pd.concat(
            [pd.DataFrame(self.spc, columns=self.wl, index=self.data.index), self.data],
            axis=1,
        )

        if string_names:
            df.columns = df.columns.map(str)

        if multiindex:
            df.columns = pd.MultiIndex.from_tuples(
                [("spc", wl) for wl in df.columns[: self.nwl]]
                + [("data", col) for col in df.columns[self.nwl :]]
            )

        return df

    # ----------------------------------------------------------------------
    # Misc.
    def sample(self, n: int, replace: bool = False) -> "SpectraFrame":
        indx = np.random.choice(self.nspc, size=n, replace=replace)
        return self[np.sort(indx), :, :, True]

    def __sizeof__(self):
        """Estimate the total memory usage"""
        return self.spc.__sizeof__() + self.data.__sizeof__() + self.wl.__sizeof__()

    # ----------------------------------------------------------------------
    # Plotting
    def _parse_string_or_vector_param(self, param: Union[str, ArrayLike]) -> pd.Series:
        if isinstance(param, str) and (param == "index"):
            return pd.Series(self.data.index)

        if isinstance(param, str) and (param in self.data.columns):
            return self.data[param]

        if len(param) == self.nspc:
            return pd.Series(param, index=self.data.index)

        raise TypeError(
            "Invalid parameter. It must be either 'index' or data column name, or"
            "array-like (i.e. np.array, list) of lenght equal to number of spectra."
        )

    def _prepare_plot_param(self, param: Union[None, str, ArrayLike]) -> pd.Series:
        if param is None:
            param = pd.Series(
                ["dummy"] * self.nspc, index=self.data.index, dtype="category"
            )
        else:
            param = self._parse_string_or_vector_param(param)

        param = (
            param.astype("category")
            .cat.add_categories("NA")
            .fillna("NA")
            .cat.remove_unused_categories()
        )

        return param

    def plot(
        self,
        rows=None,
        columns=None,
        colors=None,
        palette: Optional[list[str]] = None,
        fig=None,
        **kwargs: Any,
    ):
        # Split **kwargs
        # TODO: Either add different kw params like https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        # or infer from the name of kwarg where to put it.
        show_legend = kwargs.get("legend", colors is not None)
        sharex = kwargs.get("sharex", True)
        sharey = kwargs.get("sharey", True)

        # Convert to series all 'string or vector' params
        rows_series = self._prepare_plot_param(rows)
        cols_series = self._prepare_plot_param(columns)
        colorby_series = self._prepare_plot_param(colors)

        nrows = len(rows_series.cat.categories)
        ncols = len(cols_series.cat.categories)
        ncolors = len(colorby_series.cat.categories)

        # Prepare colors
        if palette is None:
            palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            if ncolors > len(palette):
                palette = "hsv"

        if isinstance(palette, str):
            palette = [
                rgb2hex(plt.get_cmap(palette, ncolors)(i)) for i in range(ncolors)
            ]

        cmap = dict(zip(colorby_series.cat.categories, palette[:ncolors]))
        cmap.update({"NA": "gray"})
        colors_series = colorby_series.cat.rename_categories(cmap)

        # Get the figure and the axes for plot
        if fig is None:
            fig, axs = plt.subplots(
                nrows,
                ncols,
                squeeze=False,
                sharex=sharex,
                sharey=sharey,
                layout="tight",
            )
        else:
            axs = np.array(fig.get_axes()).reshape((nrows, ncols))

        # Prepare legend lines if needed
        legend_lines = [
            Line2D([0], [0], color=c, lw=4) for c in colors_series.cat.categories
        ]

        # For each combination of row and column categories
        for i, vrow in enumerate(rows_series.cat.categories):
            for j, vcol in enumerate(cols_series.cat.categories):
                # Filter all spectra related to the current subplot
                rowfilter = np.array((rows_series == vrow) & (cols_series == vcol))
                if np.any(rowfilter):
                    self.to_pandas().iloc[rowfilter, : self.nwl].T.plot(
                        kind="line",
                        ax=axs[i, j],
                        color=colors_series[rowfilter],
                        **kwargs,
                    )

                # Add legend if needed
                if show_legend:
                    axs[i, j].legend(legend_lines, colorby_series.cat.categories)
                else:
                    axs[i, j].legend().set_visible(False)

                axs[i, j].grid()

                # For the first rows and columns set titles
                if (i == 0) and (columns is not None):
                    axs[i, j].set_title(str(vcol))
                if (j == 0) and (rows is not None):
                    axs[i, j].set_ylabel(str(vrow))

        return fig, axs

    # ----------------------------------------------------------------------
    def _to_print_dataframe(self) -> pd.DataFrame:
        print_df = self[:, :, [0, -1], True].to_pandas()
        print_df.insert(loc=1, column="...", value="...")
        return print_df

    def __str__(self) -> str:
        return self._to_print_dataframe().__str__()

    def __repr__(self) -> str:
        return self._to_print_dataframe().__repr__()
