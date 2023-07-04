from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
try:
    import fitsio
    _FITSIO = True
except ImportError:
    import astropy.io.fits
    _FITSIO = False
    # set up data type conversion
    from astropy.io.fits.column import FITS2NUMPY as format_FITS2NUMPY
    format_NUMPY2FITS = dict((v, k) for k, v in format_FITS2NUMPY.items())

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _convert_byteorder(data: NDArray) -> NDArray:
    dtype = data.dtype
    # check if the byte order matches the native order, identified by the
    # numpy dtype string representation: little endian = "<" and
    # big endian = ">"
    if dtype.str.startswith(("<", ">")):
        if sys.byteorder == "little":
            dtype = np.dtype("<" + dtype.base.str.strip("><"))
        elif sys.byteorder == "big":
            dtype = np.dtype(">" + dtype.base.str.strip("><"))
    return data.astype(dtype, casting="equiv", copy=False)


def get_FITS_format(data: NDArray) -> str:
    """
    Get a FITS data format string for given a numpy.ndarray.

    Parameters:
    -----------
    data : numpy.ndarray
        Data to generate a format string for.
    
    Returns:
    -------
    fmt_str : str
        Format string.
    """
    numpy_fmt_str = data.dtype.str.strip("<|>")
    try:
        fmt_str = format_NUMPY2FITS[numpy_fmt_str]
    except KeyError:
        raise TypeError(
            "cannot convert data of type '{:}' to FITS binary".format(
                data.dtype))
    return fmt_str


def read_fits(
    fpath: str,
    columns: Sequence[str] | None = None,
    hdu: int = 1,
    use_fitsio: bool = _FITSIO
) -> pd.DataFrame:
    """
    Read a FITS data table into a pandas.DataFrame.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    hdu : int (optional)
        Index of the extension to read from the FITS file, defaults to 1.
    columns : list of str (optional)
        Subset of columns to read from the table, defaults to all.
    
    Returns:
    -------
    df : pandas.DataFrame
        Table data converted to a DataFrame instance.
    """
    # load the FITS data
    if use_fitsio:
        fits = fitsio.FITS(fpath)
        try:
            if columns is None:
                data = fits[hdu][:]
            else:
                data = fits[hdu][columns][:]
        finally:
            fits.close()

        # construct the data frame
        coldata = {}
        for colname, (dt, _) in data.dtype.fields.items():
            if len(dt.shape) > 0:
                warnings.warn(
                    "dropping multidimensional column '{:}'".format(colname))
            else:
                coldata[colname] = _convert_byteorder(data[colname])
        dataframe = pd.DataFrame(coldata)

    else:
        with astropy.io.fits.open(fpath) as fits:
            if columns is None:
                # probably better to read data at once but I did not find a good
                # way to identify and convert string columns correctly to fixed
                # width
                columns = fits[hdu].data.dtype.names
            nrows = fits[hdu].data.shape[0]
            vals = [fits[hdu].data[c] for c in columns]
            dtype = np.dtype([
                (c, v.dtype.str) for c, v in zip(columns, vals)])
            data = np.empty(nrows, dtype=dtype)
            for c, v in zip(columns, vals):
                data[c] = v

        # construct the data frame
        coldata = {}
        for colname, (dt, _) in data.dtype.fields.items():
            if dt.kind == "O":
                coldata[colname] = np.array([
                    "".join(chars) for chars in data[colname]]).astype("|S")
            elif len(dt.shape) > 0:
                warnings.warn(
                    "dropping multidimensional column '{:}'".format(colname))
            else:
                coldata[colname] = _convert_byteorder(data[colname])
        dataframe = pd.DataFrame(coldata)

    # convert string types to fixed width (pandas default are py str objects)
    for colname, values in coldata.items():
        if values.dtype.kind in "SU":
            # FITS does not support unicode, use bytes array instead
            dataframe[colname] = values.astype("|S")
    return dataframe


def read_auto(
    fpath: str,
    columns: Sequence[str] | None = None,
    ext: str | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a file by guessing its type from the extension. Standard parameters
    are used for the pandas.read_xxx() method.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    ext : str (optional)
        Manual overwrite for the file extension.
    columns : list of str (optional)
        Subset of columns to read from the table (if supported), defaults to
        all.
    **kwargs
        Passed on to the pandas.read_xxx() method.
    
    Returns:
    -------
    df : pandas.DataFrame
        Table data read as DataFrame.
    """
    # parse the extension
    if ext is None:
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
    # parse the columns parameter
    if columns is not None:
        if ext in (".json", ".html", ".pkl"):
            warnings.warn(
                f"column subsets are not supported for extension '{ext}'")
        elif ext == ".csv":
            kwargs["usecols"] = columns
        else:
            kwargs["columns"] = columns
    # read data
    if ext in (".csv",):
        return pd.read_csv(fpath, **kwargs)
    elif ext in (".json",):
        return pd.read_json(fpath, **kwargs)
    elif ext in (".html",):
        return pd.read_html(fpath, **kwargs)
    elif ext in (".hdf5", ".h5"):
        return pd.read_hdf(fpath, **kwargs)
    elif ext in (".pqt", ".parquet"):
        return pd.read_parquet(fpath, **kwargs)
    elif ext in (".feather"):
        return pd.read_feather(fpath, **kwargs)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(fpath, **kwargs)
    elif ext in (".fits", ".cat"):
        return read_fits(fpath, **kwargs)
    else:
        raise ValueError("unrecognized file extesion '{:}'".format(ext))


def to_fits(
    df: pd.DataFrame,
    fpath: str,
    use_fitsio: bool = _FITSIO
) -> None:
    """
    Write a pandas.DataFrame as FITS table file.

    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame to write as FITS table.
    fpath : str
        Path to the FITS file.
    """
    # load the FITS data
    if use_fitsio:
        dtype = np.dtype(list(df.dtypes.items()))
        array = np.empty(len(df), dtype=dtype)
        for column in df.columns:
            array[column] = df[column]
        if os.path.exists(fpath):
            os.remove(fpath)
        with fitsio.FITS(fpath, "rw") as fits:
            fits.write(array)
    else:
        columns = []
        for col in df.columns:
            try:
                coldef = astropy.io.fits.Column(
                    name=col,
                    format=get_FITS_format(df[col]),
                    array=df[col])
                columns.append(coldef)
            except TypeError:
                warnings.warn(
                    "dropping column '{:}' with unsuppored type '{:}'".format(
                        col, df[col].dtype))
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns)
        hdu.writeto(fpath, overwrite=True)


def to_auto(
    df: pd.DataFrame,
    fpath: str,
    ext: int | None = None,
    **kwargs
) -> None:
    """
    Write a file to a file format using standard parameters for the
    pandas.to_xxx() method.

    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame to write.
    fpath : str
        Path to the FITS file.
    ext : str
        Manual overwrite for the file extension.
    **kwargs
        Passed on to the pandas.to_xxx() method.
    """
    if ext is None:
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
    if ext in (".csv",):
        df.to_csv(fpath, **kwargs)
    elif ext in (".json",):
        df.to_json(fpath, **kwargs)
    elif ext in (".html",):
        df.to_html(fpath, **kwargs)
    elif ext in (".hdf5", ".h5"):
        df.to_hdf(fpath, **kwargs)
    elif ext in (".pqt", ".parquet"):
        df.to_parquet(fpath, **kwargs)
    elif ext in (".feather"):
        df.to_feather(fpath, **kwargs)
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(fpath, **kwargs)
    elif ext in (".fits", ".cat"):
        to_fits(df, fpath, **kwargs)
    else:
        raise ValueError("unrecognized file extesion '{:}'".format(ext))
