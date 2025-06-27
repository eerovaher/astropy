from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def interval_overlap_length(i1: tuple[float, float], i2: tuple[float, float]) -> float:
    """Compute the length of overlap of two intervals.

    Parameters
    ----------
    i1, i2 : (float, float)
        The two intervals, (interval 1, interval 2).

    Returns
    -------
    l : float
        The length of the overlap between the two intervals.

    """
    (a, b) = i1
    (c, d) = i2
    if a < c:
        if b < c:
            return 0.0
        elif b < d:
            return b - c
        else:
            return d - c
    elif a < d:
        if b < d:
            return b - a
        else:
            return d - a
    else:
        return 0


def histogram_intervals(
    n: int, breaks: Sequence[float], totals: Sequence[float]
) -> NDArray[np.float64]:
    """Histogram of a piecewise-constant weight function.

    This function takes a piecewise-constant weight function and
    computes the average weight in each histogram bin.

    Parameters
    ----------
    n : int
        The number of bins
    breaks : (N,) array of float
        Endpoints of the intervals in the PDF
    totals : (N-1,) array of float
        Probability densities in each bin

    Returns
    -------
    h : array of float
        The average weight for each bin

    """
    h = np.zeros(n)
    start = breaks[0]
    for i in range(len(totals)):
        end = breaks[i + 1]
        for j in range(n):
            ol = interval_overlap_length((float(j) / n, float(j + 1) / n), (start, end))
            h[j] += ol / (1.0 / n) * totals[i]
        start = end

    return h
