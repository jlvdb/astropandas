# astropandas

Tools to expand on pandas functionality for astronomical applications.

## Features

- directly read and write FITS table data, see `astropandas.read_fits` and `astropandas.to_fits`
- match based on sky coordinates, see `astropandas.match`

## Examples

Reading and writing FITS table files:
```python
import astropandas as apd

data = apd.read_fits("myfile.fits", hdu=1)
# manipulate data ...
apd.to_fits(data, "mynewfile.fits")
```

Matching on angular coordinates:
```python
import astropandas as apd

data1 = apd.read_fits("myfile1.fits")
data2 = apd.read_fits("myfile2.fits")

# Match within one arcsecond on right ascension (ra) and declination (dec),#
# labeled "RA" and "DEC" in the data.
# Here we assume that the column names are the same in both catalogues,
# otherwise specify with left_ra=, right_ra=, etc.
apd.match(
    left=data1, right=data2,
    ra="RA", dec="DEC",
    threshold=1/3600)
# If no threshold is provided, it is computed automatically by finding the
# distance at which the number of matches is almost stationary.
```
