import numpy as np

"""Run length algorithms.

Need to benchmark and adapt for xarray. 

"""


def rle(arr):
    """Return the length, starting position and value of consecutive identical values.

    Parameters
    ----------
    arr : sequence
      Array of values to be parsed.

    Returns
    -------
    (values, run lengths, start positions)
    values : np.array
      The values taken by arr over each run.
    run lengths : np.array
      The length of each run.
    start position : np.array
      The starting index of each run.

    Examples
    --------
    >>> a = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    >>> rle(a)
    (array([1, 2, 3]), array([2, 4, 6]), array([0, 2, 6]))

    """
    ia = np.asarray(arr)
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = np.array(ia[1:] != ia[:-1])         # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)       # must include last element position
        rl = np.diff(np.append(-1, i))          # run lengths
        pos = np.cumsum(np.append(0, rl))[:-1]  # positions
        return ia[i], rl, pos


def windowed_run_count(arr, window):
    """Return the number of consecutive true values in array for runs at least as long as given duration.

    Parameters
    ----------
    arr : bool array
      Input array.
    window : int
      Minimum duration of consecutive run to accumulate values.

    Returns
    -------
    int
      Total number of true values part of a consecutive run at least `window` long.
    """
    v, rl = rle(arr)[:2]
    return np.where(v * rl >= window, rl, 0).sum()


def longest_run(arr):
    """Return the length of the longest consecutive run of identical values.

    Parameters
    ----------
    arr : bool array
      Input array.

    Returns
    -------
    int
      Length of longest run.
    """
    v, rl = rle(arr)[:2]
    return np.where(v, rl, 0).max()
