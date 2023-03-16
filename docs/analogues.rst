Spatial Analogues
=================

Spatial analogues are maps showing which areas have a present-day climate that is analogous to the future climate of a
given place. This type of map can be useful for climate adaptation to see how regions are coping today under
specific climate conditions. For example, officials from a city located in a temperate region that may be expecting more
heatwaves in the future can learn from the experience of another city where heatwaves are a common occurrence,
leading to more proactive intervention plans to better deal with new climate conditions.

Spatial analogues are estimated by comparing the distribution of climate indices computed at the target location over
the future period with the distribution of the same climate indices computed over a reference period for multiple
candidate regions. A number of methodological choices thus enter the computation:

    - Climate indices of interest,
    - Metrics measuring the difference between the distributions of indices,
    - Reference data from which to compute the base indices,
    - A future climate scenario to compute the target indices.

The climate indices chosen to compute the spatial analogues are usually annual values of indices relevant to the
intended audience of these maps. For example, in the case of the wine grape industry, the climate indices examined could
include the length of the frost-free season, growing degree-days, annual winter minimum temperature and annual number of
very cold days :cite:p:`roy_probabilistic_2017`.

See :ref:`notebooks/analogs:Spatial Analogues examples`.

Methods to compute the (dis)similarity between samples
------------------------------------------------------
This module implements all methods described in :cite:cts:`grenier_assessment_2013` to measure the dissimilarity between
two samples, as well as the Székely-Rizzo energy distance. Some of these algorithms can be used to test whether two samples
have been drawn from the same distribution. Here, they are used in finding areas with analogue climate conditions to a
target climate:

 * Standardized Euclidean distance
 * Nearest Neighbour distance
 * Zech-Aslan energy statistic
 * Székely-Rizzo energy distance
 * Friedman-Rafsky runs statistic
 * Kolmogorov-Smirnov statistic
 * Kullback-Leibler divergence

All methods accept arrays, the first is the reference (n, D) and the second is the candidate (m, D). Where the climate
indicators vary along D and the distribution dimension along n or m. All methods output a single float. See their
documentation in :ref:`analogues:Analogues Metrics API`.

.. warning::

   Some methods are scale-invariant and others are not. This is indicated in the docstring
   of the methods as it can change the results significantly. In most cases, scale-invariance
   is desirable and inputs may need to be scaled beforehand for scale-dependent methods.

.. rubric:: References

:cite:cts:`roy_probabilistic_2017`
:cite:cts:`grenier_assessment_2013`

Analogues Metrics API
---------------------

See: :ref:`spatial-analogues-api`

Analogues Developer Functions
-----------------------------

See: :ref:`spatial-analogues-developer-api`
