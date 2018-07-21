import numpy as np
import pandas as pd

from itertools import product
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean


class Prickle(object):
    """
    Args:
        samples (`pandas.DataFrame`): Shape [`i`, `j`]. Prickle plot will
            contain `i` rows and `j` columns. Each element in `samples` is
            `array-like`, shape [2, `m`]. If there is no data for element, its
            value should be `nan`.
        zero (`array-like`): Shape [2, ].
    """
    def __init__(self, samples, zero):
        self.samples = samples
        self.zero = zero
        self.nrows = samples.shape[0]
        self.ncols = samples.shape[1]
        self.rows = range(self.nrows)
        self.cols = range(self.ncols)
        self.ij = product(self.rows, self.cols)

    def plot_dots(self, **kwds):
        """Plot dots showing zero values for all elements in `samples` that
        have `m` > 0.

        Args:
            **kwds: Keyword arguments passed to ax.scatter().

        Returns:
            `matplotlib.axes.Axes`
        """
        ax = plt.gca()
        dots = np.argwhere(self.samples.notnull().values)
        s = kwds.pop('s', 10)
        c = kwds.pop('c', 'black')
        ax.scatter(dots[:, 1], dots[:, 0], s=s, c=c, **kwds)
        return ax

    def _make_segments(self):
        """Segments are the prickle lines in the plot.

        Segments are defined by their start and end coordinates. Segments are
        attached as an attribute on self.

        Attributes:
            segments (`list`): Each element is an np.array, shape (2, 2):
                np.array([
                    [x0, y0],
                    [x1, y1]
                ])
        """
        self.segments = []
        append = self.segments.append

        for i, j in self.ij:
            element = self.samples.iloc[i, j]

            # Check element is array / list / tuple
            if hasattr(element, "__len__"):

                vectors = np.array(element) - self.zero.values
                x0, y0 = j, i

                for vector in vectors:
                    x1 = x0 + vector[0]
                    y1 = y0 + vector[1]
                    append(np.array([[x0, y0], [x1, y1]]))

            elif pd.notnull(element):
                warn(
                    "Unhandled element at samples.loc[{}, {}] "
                    "It is not null, and it could not be plotted".format(i, j)
                )

    def _segment_lengths(self):
        """Compute the lengths of all segments.

        Returns:
            np.array: Shape (len(self.segments), ). The lengths of all
                segments.
        """
        if not hasattr(self, 'segments'):
            self._make_segments()

        return np.array(
            [euclidean(self.segments[i][0], self.segments[i][1])
             for i in range(len(self.segments))])

    def hist(self, **kwds):
        """Plot a histogram of the lengths of each prickle line.

        Arguments:
            **kwds: Keyword arguments passed to plt.hist().
        """
        plt.hist(self._segment_lengths(), **kwds)

    def plot_prickles(self, **kwds):
        """Plot prickles.

        Args:
            **kwds: Keyword arguments passed to
                `matplotlib.collections.LineCollection`.
                `label` is an additional keyword argument. If it is a `str`
                then proxy artists are appended to a list, self.handles, with
                label=label. Draw the legend using: ax.legend(handles=handles).

        Returns:
            `matplotlib.axes.Axes`
        """
        if not hasattr(self, 'segments'):
            self._make_segments()

        ax = plt.gca()
        linewidths = kwds.pop('linewidths', 1)
        colors = kwds.pop('colors', 'black')
        label = kwds.pop('label', None)

        if label:
            lw_is_scalar = isinstance(linewidths, (int, float))
            colors_is_str = isinstance(colors, str)
            if not lw_is_scalar or not colors_is_str:
                raise NotImplementedError(
                    "Cannot set label when multiple colors and/or linewidths "
                    "are used.")

            if not hasattr(self, 'handles'):
                self.handles = list()

            self.handles.append(
                matplotlib.lines.Line2D(
                    xdata=[],
                    ydata=[],
                    color=colors,
                    linewidth=linewidths,
                    label=label))

        lc = matplotlib.collections.LineCollection(
            segments=self.segments,
            linewidths=linewidths,
            colors=colors,
            **kwds)

        ax.add_artist(lc)
        return ax

    def plot(self, pad=1, dot_kwds={}, prickle_kwds={}):
        """Draw the prickle plot.

        Args:
            pad (`number`): Ax padding.
            dot_kwds (`dict`): Keywords to pass to Prickle.plot_dots().
            prickle_kwds (`dict`): Keywords to pass to Prickle.plot_prickles().

        Returns:
            `matplotlib.axes.Axes`
        """
        self.plot_dots(**dot_kwds)
        self.plot_prickles(**prickle_kwds)
        ax = plt.gca()
        ax.set_xticks(self.cols)
        ax.set_yticks(self.rows)
        ax.set_xticklabels(self.samples.columns)
        ax.set_yticklabels(self.samples.index)
        ax.set_aspect(1)
        ax.set_xlim(-pad, self.ncols + pad - 1)
        ax.set_ylim(-pad, self.nrows + pad - 1)
        return ax
