import os
import sys
import warnings

import numpy as np
import pandas as pd
try:
    import fitsio
    _FITSIO = True
except ImportError:
    import astropy.io
    _FITSIO = False


def _convert_byteorder(data):
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


def read_fits(fpath, cols=None, hdu=1):
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
        df : pandas.DataFrame
            Table data converted to a DataFrame instance.
    """
    # load the FITS data
    if _FITSIO:
        fits = fitsio.FITS(fpath)
        if cols is None:
            data = fits[hdu][:]
        else:
            data = fits[hdu][cols][:]
        fits.close()
    else:
        with astropy.io.fits.open(fpath) as fits:
            if cols is None:
                data = fits[hdu].data
            else:
                data = fits[hdu][cols].data
    # construct the data frame
    coldata = {}
    for colname, (dt, _) in data.dtype.fields.items():
        if len(dt.shape) > 0:
            warnings.warn(
                "dropping multidimensional column '{:}'".format(colname))
        else:
            coldata[colname] = _convert_byteorder(data[colname])
    return pd.DataFrame(coldata)


def read_auto(fpath, ext=None, **kwargs):
    """
    Read a file by guessing its type from the extension. Standard parameters
    are used for the pandas.read_xxx() method.

    Parameters:
    -----------
    fpath : str
        Path to the FITS file.
    ext : str
        Manual overwrite for the file extension
    **kwargs
        Passed on to the pandas.read_xxx() method
    
    Returns:
        df : pandas.DataFrame
            Table data read as DataFrame.
    """
    if ext is None:
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
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
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(fpath, **kwargs)
    elif ext in (".fits", ".cat"):
        return read_fits(fpath, **kwargs)
    else:
        raise ValueError("unrecognized file extesion '{:}'".format(ext))


def to_fits(df, fpath):
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
    if _FITSIO:
        dtype = np.dtype(list(df.dtypes.items()))
        array = np.empty(len(df), dtype=dtype)
        for column in df.columns:
            array[column] = df[column]
        if os.path.exists(fpath):
            os.remove(fpath)
        with fitsio.FITS(fpath, "rw") as fits:
            fits.write(array)
    else:
        columns = [
            astropy.io.fits.Column(name=col, format='E', array=df[col])
            for col in df.columns]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns)
        hdu.writeto(fpath, overwrite=True)


def to_auto(df, fpath, ext=None, **kwargs):
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
        Manual overwrite for the file extension
    **kwargs
        Passed on to the pandas.to_xxx() method
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
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(fpath, **kwargs)
    elif ext in (".fits", ".cat"):
        to_fits(df, fpath, **kwargs)
    else:
        raise ValueError("unrecognized file extesion '{:}'".format(ext))
