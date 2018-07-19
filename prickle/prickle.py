import numpy as np
import pandas as pd

from itertools import product
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

class Prickle(object):
    """
    Args:
        samples (`:py:class:pandas.DataFrame`): Shape [`i`, `j`]. Prickle plot
            will contain `i` rows and `j` columns. Each element in `samples`
            is `array-like`, shape [2, `m`]. If there is no data for element,
            its value should be `nan`.
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
            **kwds (Keyword arguments): Passed to ax.scatter().

        Returns:
            `matplotlib.axes.Axes`
        """
        ax = plt.gca()
        dots = np.argwhere(self.samples.notnull().values)
        s = kwds.pop('s', 10)
        c = kwds.pop('c', 'black')
        ax.scatter(dots[:, 1], dots[:, 0], s=s, c=c, **kwds)
        return ax

    def plot_prickles(self, **kwds):
        """Plot prickles.

        Args:
            **kwds (Keyword arguments): Passed to
                `matplotlib.collections.LineCollection`.

        Returns:
            `matplotlib.axes.Axes`
        """
        segments = []
        append = segments.append

        for i, j in self.ij:
            element = self.samples.iloc[i, j]

            # Check element is array / list / tuple
            if hasattr(element, "__len__"):

                vectors = np.array(element) - self.zero.values
                x0, y0 = j, i

                for vector in vectors:
                    x1 = x0 + vector[0]
                    y1 = y0 + vector[1]
                    append([[x0, y0], [x1, y1]])

            elif pd.notnull(element):
                warn("Odd element in df at samples.loc[{}, {}] "
                     "It is not null, and it could not be plotted".format(i, j))

        ax = plt.gca()
        linewidths = kwds.pop('linewidths', 1)
        colors = kwds.pop('colors', 'black')
        lc = LineCollection(
            segments=segments, linewidths=linewidths, colors=colors, **kwds)
        ax.add_artist(lc)
        return ax

    def plot(self, pad=1, dot_kwds={}, prickle_kwds={}):
        """Draw the prickle plot.

        Args:
            pad (Scalar): Ax padding.
            dot_kwds (Dict): Keyword to pass to Prickle.plot_dots()
            prickle_kwds (Dict): Keyword to pass to Prickle.plot_prickles()
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
