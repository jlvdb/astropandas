import warnings

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


class SphericalKDTree(object):
    """
    SphericalKDTree(ra, dec, leaf_size=16)

    A binary search tree based on scipy.spatial.cKDTree that works with
    celestial coordinates to find nearest neighbours. Data is internally
    represented on a unit-sphere in three dimensions (x, y, z).

    Parameters
    ----------
    ra : array_like
        List of right ascensions in degrees.
    dec : array_like
        List of declinations in degrees.
    leafsize : int
        The number of points at which the algorithm switches over to
        brute-force.
    """

    def __init__(self, ra, dec, leafsize=16):
        if len(ra) != len(dec):
            raise ValueError("'ra' and 'dec' must have the same length")
        # convert angular coordinates to 3D points on unit sphere
        self.data = np.transpose([ra, dec])
        pos_sphere = self._position_sky2sphere(ra, dec)
        self._tree = scipy.spatial.cKDTree(pos_sphere, leafsize)

    @staticmethod
    def _position_sky2sphere(ra, dec):
        """
        Maps celestial coordinates onto a unit-sphere in three dimensions
        (x, y, z).

        Parameters
        ----------
        ra : float or array_like
            Single or list of right ascensions in degrees.
        dec: float or array_like
            Single or list of declinations in degrees.

        Returns
        -------
        pos_sphere : array like
            Data points (x, y, z) representing input points on the unit-sphere,
            shape of output is (3,) for a single input point or (N, 3) for a
            set of N input points.
        """
        ras_rad = np.atleast_1d(np.deg2rad(ra))
        decs_rad = np.atleast_1d(np.deg2rad(dec))
        pos_sphere = np.empty((len(ras_rad), 3))
        cos_decs = np.cos(decs_rad)
        pos_sphere[:, 0] = np.cos(ras_rad) * cos_decs
        pos_sphere[:, 1] = np.sin(ras_rad) * cos_decs
        pos_sphere[:, 2] = np.sin(decs_rad)
        return np.squeeze(pos_sphere)

    @staticmethod
    def _distance_sky2sphere(dist_sky):
        """
        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.

        Parameters
        ----------
        dist_sky : float or array_like
            Single or list of separations in celestial coordinates.

        Returns
        -------
        dist_sphere : float or array_like
            Celestial separation converted to (x, y, z) Euclidean distance.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere):
        """
        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.

        Parameters
        ----------
        dist_sphere : float or array_like
            Single or list of Euclidean distances in (x, y, z) space.

        Returns
        -------
        dist_sky : float or array_like
            Euclidean distance converted to celestial angular separation.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def query(self, ra, dec, k=1, distance_upper_bound=np.inf, workers=1):
        """
        Find all data points within an angular aperture r around a reference
        point with coordiantes (RA, DEC) obeying the spherical geometry.

        Parameters
        ----------
        ra : float or array_like
            Single or list of right ascensions in degrees.
        dec : float or array_like
            Single or list of declinations in degrees.
        k : int
            Number of nearest neighbours to query.
        distance_upper_bound : float
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        workers : int
            Number of workers to use for parallel processing. If -1 is given
            all CPU threads are used.

        Returns
        -------
        d : array_like of float
            Angular separation of nearest neighbors in degrees. Dimensions are
            squeezed if input ra/dec are scalar. Missing neighbors are indicated
            with infinite distances.
        i : array_like of int
            Positional indices of nearest neighbors in self.data. Dimensions are
            squeezed if input ra/dec are scalar. Missing neighbors are indicated
            with length of self.data.
        """
        points = self._position_sky2sphere(ra, dec)
        distance, i = self._tree.query(
            points, k, distance_upper_bound=self._distance_sky2sphere(
                distance_upper_bound),
            workers=workers)
        # convert the distance to angular separation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = self._distance_sphere2sky(distance)
        return d, i

    def count_neighbors(self, other, r, cumulative=True):
        """
        Count the number of pairs between data and other within a give angular
        separation.

        Parameters
        ----------
        other : float or array_like
            Single or list of right ascensions in degrees.
        r: float or array_like
            The angular separation to produce a count for. Multiple radii are
            searched with a single tree traversal. If the count is non-
            cumulative (cumulative=False), r defines the edges of the bins, and
            must be non-decreasing.
        cumulative : bool
            Whether the returned counts are cumulative. When cumulative is set
            to False the algorithm is optimized to work with a large number of
            bins (>10) specified by r. When cumulative is set to True, the
            algorithm is optimized to work with a small number of r. 

        Returns
        -------
        result : int or array_like
            The number of pairs, if cumulative is False, result[i] contains the
            counts with (-inf if i == 0 else r[i-1]) < R <= r[i].
        """
        if not isinstance(other, SphericalKDTree):
            raise TypeError("'other' must be an instance of 'SphericalKDTree'")
        result = self._tree.count_neighbors(
            other._tree, r=self._distance_sky2sphere(r), cumulative=cumulative)
        return result


class MatchInfo:

    def __init__(self, separations, counts, threshold):
        self.separations = separations
        self.counts = counts
        self.threshold = threshold
        self.distances = None

    def set_distances(self, delta_ra, delta_dec):
        self.distances = np.transpose([delta_ra, delta_dec])

    def offset(self):
        return np.nanmean(self.distances, axis=0)

    def scatter(self):
        return np.nanstd(self.distances, axis=0)

    def plot_threshold(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.separations * 3600.0, self.counts)
        if np.isfinite(self.threshold):
            thresh = self.threshold * 3600.0
            ax.axvline(thresh, color="k")
            ax.annotate(
                "thresh: {:.3f} arcsec".format(thresh),
                xy=(thresh, np.mean(ax.get_ylim())),
                xycoords="data", va="center", ha="right", rotation=90)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Point separation / arcsec")
        ax.set_ylabel("Frequency")

    def plot_offset(self, ax=None, sigmas=3, aspect=False):
        mask = np.isfinite(self.distances).all(axis=1)
        delta_ra, delta_dec = np.transpose(self.distances[mask] * 3600.0)
        offset = self.offset() * 3600.0
        scatter = self.scatter() * 3600.0
        ax_range = sigmas * max(scatter)
        # make the plot
        if ax is None:
            ax = plt.gca()
        ax.hist2d(
            delta_ra, delta_dec, bins=np.linspace(-ax_range, ax_range, 51),
            cmap="Blues")
        # mark the offset and scatter
        ax.scatter(*offset, color="C3", marker="+")
        ax.add_patch(matplotlib.patches.Ellipse(
            tuple(offset), *(2*scatter), color="C3", ls="--", fill=False))
        ax.grid(alpha=0.3)
        ax.annotate(
            "max=%.3f" % delta_ra.max(), (0.5, 0.05), xycoords="axes fraction",
            va="center", ha="center")
        ax.annotate(
            "max=%.3f" % delta_dec.max(), (0.05, 0.5), xycoords="axes fraction",
            va="center", ha="center", rotation=90)
        ax.set_xlim(-ax_range, ax_range)
        ax.set_ylim(-ax_range, ax_range)
        if aspect:
            ax.set_aspect("equal")
        ax.set_xlabel(r"$\Delta$RA / arcsec")
        ax.set_ylabel(r"$\Delta$DEC / arcsec")

    def plot(self):
        fig, (axt, axo) = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_threshold(axt)
        self.plot_offset(axo, aspect=True)
        fig.tight_layout()


class Matcher:

    def __init__(
            self, left, right, left_ra, right_ra, left_dec, right_dec):
        self.left = left
        self.right = right
        # store the column names
        self.left_ra, self.right_ra = left_ra, right_ra
        self.left_dec, self.right_dec = (left_dec, right_dec)
        # build the binary search tree for neighbour matching
        self.tree_left = SphericalKDTree(
            self.left[left_ra].to_numpy(), self.left[left_dec].to_numpy())
        self.tree_right = SphericalKDTree(
            self.right[right_ra].to_numpy(), self.right[right_dec].to_numpy())

    def auto_threshold(self, threshold=None):
        # build a quadratially spaced grid to search for an automatic threshold
        separations = np.linspace(
            np.sqrt(1.4e-5),  # about 0.05 arcsec
            np.sqrt(1.4e-3),  # about 5 arcsec
            100)**2
        counts = self.tree_left.count_neighbors(self.tree_right, r=separations)
        if threshold is None:
            # find separation that minimises the rate of change in the pair count
            idx_minimum = np.argmin(np.diff(counts))
            # take the mean with the neighboring threshold grid point
            threshold = separations[idx_minimum:idx_minimum+2].mean()
        return MatchInfo(separations, counts, threshold)

    def match(
            self, how="inner", threshold=None, sort=False,
            suffixes=("_x", "_y"), copy=True, indicator=False, workers=1):
        # create matching indices, initialised with no overlap
        left_idx = np.arange(len(self.left))
        right_idx = np.arange(len(self.right)) + len(self.left)
        info = self.auto_threshold(threshold)
        # idx_match: for each object in right -> index of nearest object in left
        dist, idx_match = self.tree_left.query(
            *self.tree_right.data.T, k=1, distance_upper_bound=info.threshold,
            workers=workers)
        match_mask = np.isfinite(dist)
        if not match_mask.any():
            raise ValueError(
                "no match found within {:.1f} arcsec".format(info.threshold))
        idx_match_left = dict(zip(
            right_idx.compress(match_mask),
            left_idx[idx_match.compress(match_mask)]))
        # idx_match: for each object in left -> index of nearest object in right
        dist, idx_match = self.tree_right.query(
            *self.tree_left.data.T, k=1, distance_upper_bound=info.threshold,
            workers=workers)
        match_mask = np.isfinite(dist)
        if not match_mask.any():
            raise ValueError(
                "no match found within {:.1f} arcsec".format(info.threshold))
        idx_match_right = dict(zip(
            left_idx.compress(match_mask),
            right_idx[idx_match.compress(match_mask)]))
        # find the symmetric matches
        index_left = left_idx.copy()
        n = 0
        for i_left, i_right in idx_match_right.items():
            # check if the match is symmetric (each rights closest partner)
            if i_left == idx_match_left.get(i_right, -1):
                index_left[i_left] = i_right
                n += 1
        # insert the indices temporarily for the pandas join method
        temp_key = "__spatial_match"
        self.left[temp_key] = index_left
        self.right[temp_key] = right_idx
        try:
            # do a simple 1-D index match on the left and right index
            merged = self.left.merge(
                self.right, how=how, on=temp_key, sort=sort, suffixes=suffixes,
                copy=copy, indicator=indicator)
        finally:  # always remove the temporary columns
            self.left.pop(temp_key)
            self.right.pop(temp_key)
        try:
            merged.pop(temp_key)
        except KeyError:
            pass
        # get the columns names in the merged data frame
        left_ra = self.left_ra if self.left_ra in merged else \
                  self.left_ra + suffixes[0]
        right_ra = self.right_ra if self.right_ra in merged else \
                   self.right_ra + suffixes[1]
        left_dec = self.left_dec if self.left_dec in merged else \
                   self.left_dec + suffixes[0]
        right_dec = self.right_dec if self.right_dec in merged else \
                    self.right_dec + suffixes[1]
        # collect distances between matches
        info.set_distances(
            merged[left_ra] - merged[right_ra],
            merged[left_dec] - merged[right_dec])
        # merge the RA/Dec columns used for matching
        iter_cols = zip([left_ra, left_dec], [right_ra, right_dec])
        for left_name, right_name in iter_cols:
            # correct the name if a suffix was appended to the orginal names
            if left_name not in merged:
                left_name += suffixes[0]
            if right_name not in merged:
                right_name += suffixes[1]
            # remove the column data from the merged frame
            insert_idx = merged.columns.get_loc(left_name)
            left_data = merged.pop(left_name)
            right_data = merged.pop(right_name)
            # keep values from self if they are set, rightwise right and
            # reinsert under original name in left frame
            data = np.where(np.isnan(left_data), right_data, left_data)
            merged.insert(
                insert_idx, left_name.replace(suffixes[0], ""), data)
        return merged, info


def match(
        left, right, how="inner", threshold=None,
        ra=None, left_ra=None, right_ra=None,
        dec=None, left_dec=None, right_dec=None,
        sort=False, suffixes=("_x", "_y"), copy=True,
        indicator=False, workers=1):
    # collect the key for Right Ascension
    if ra is not None:
        if ra not in left or ra not in right:
            raise KeyError(ra)
        left_ra = ra
        right_ra = ra
    else:
        if left_ra is None or right_ra is None:
            raise ValueError("Right Ascension keys must be specified")
        if left_ra not in left:
            raise KeyError(left_ra)
        if right_ra not in right:
            raise KeyError(right_ra)
    # collect the key for Declination
    if dec is not None:
        if dec not in left or dec not in right:
            raise KeyError(dec)
        left_dec = dec
        right_dec = dec
    else:
        if left_dec is None or right_dec is None:
            raise ValueError("Declination keys must be specified")
        if left_dec not in left:
            raise KeyError(left_dec)
        if right_dec not in right:
            raise KeyError(right_dec)
    # perform the matching
    matcher = Matcher(left, right, left_ra, right_ra, left_dec, right_dec)
    matched, info = matcher.match(
        how=how, threshold=threshold, sort=sort,
        suffixes=suffixes, workers=workers, copy=copy, indicator=indicator)
    return matched, info
