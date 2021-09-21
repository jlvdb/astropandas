import sys

import numpy as np
import pandas as pd
try:
    import fitsio
    _FITSIO = True
except ImportError:
    import astropy.io
    _FITSIO = False

from .match import Matcher


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
                data = fits[hdu]
            else:
                data = fits[hdu][cols]
    # construct the data frame
    df = pd.DataFrame(data={
        colname: _convert_byteorder(data[colname])
        for colname, _ in data.dtype.fields.items()})
    return df


class DataFrame(pd.DataFrame):

    def match(
            self, right, how="inner", threshold=None,
            ra=None, left_ra=None, right_ra=None,
            dec=None, left_dec=None, right_dec=None,
            sort=False, suffixes=("_x", "_y"), copy=True,
            indicator=False, workers=1):
        # collect the key for Right Ascension
        if ra is not None:
            if ra not in self or ra not in right:
                raise KeyError(ra)
            left_ra = ra
            right_ra = ra
        else:
            if left_ra is None or right_ra is None:
                raise ValueError("Right Ascension keys must be specified")
            if left_ra not in self:
                raise KeyError(left_ra)
            if right_ra not in right:
                raise KeyError(right_ra)
        # collect the key for Declination
        if dec is not None:
            if dec not in self or dec not in right:
                raise KeyError(dec)
            left_dec = dec
            right_dec = dec
        else:
            if left_dec is None or right_dec is None:
                raise ValueError("Declination keys must be specified")
            if left_dec not in self:
                raise KeyError(left_dec)
            if right_dec not in right:
                raise KeyError(right_dec)
        # perform the matching
        matcher = Matcher(self, right, left_ra, right_ra, left_dec, right_dec)
        match, info = matcher.match(
            self, how=how, threshold=threshold, sort=sort,
            suffixes=suffixes, workers=workers, copy=copy, indicator=indicator)
        return match, info


    def to_fits(self, fpath):
        """
        Write a pandas.DataFrame as FITS table file.

        Parameters:
        -----------
        fpath : str
            Path to the FITS file.
        """
        # load the FITS data
        if _FITSIO:
            dtype = np.dtype(list(self.dtypes.items()))
            array = np.empty(len(self), dtype=dtype)
            for column in self.columns:
                array[column] = self[column]
            with fitsio.FITS(fpath, "rw") as fits:
                fits.write(array)
        else:
            columns = [
                astropy.io.fits.Column(name=col, array=self[col])
                for col in self.columns]
            hdu = astropy.io.fits.BinTableHDU.from_columns(columns)
            hdu.writeto(fpath)
