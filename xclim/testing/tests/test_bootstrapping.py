import numpy as np

from xclim.core.calendar import percentile_doy
from xclim.indices import days_over_precip_thresh, tg90p


def ar1(alpha, n):
    """Return random AR1 DataArray."""

    # White noise
    wn = np.random.randn(n - 1) * np.sqrt(1 - alpha ** 2)

    # Autoregressive series of order 1
    out = np.empty(n)
    out[0] = np.random.randn()
    for i, w in enumerate(wn):
        out[i + 1] = alpha * out[i] + w

    return out


def test_bootstrap_tas(tas_series):
    n = int(60 * 365.25)
    alpha = 0.8
    tas = tas_series(ar1(alpha, n), start="2000-01-01")
    in_base_slice = slice("2000-01-01", "2029-12-31")
    out_base_slice = slice("2030-01-01", "2059-12-31")
    per = percentile_doy(tas.sel(time=in_base_slice), per=90)

    no_bootstrap = tg90p(tas, per, freq="MS", bootstrap=False)
    biaised_period = no_bootstrap.sel(time=(in_base_slice))
    no_bs_out_base = no_bootstrap.sel(time=(out_base_slice))

    bootstrap = tg90p(tas, per, freq="MS", bootstrap=True)
    corrected_period = bootstrap.sel(time=(in_base_slice))
    bs_out_base = bootstrap.sel(time=(out_base_slice))

    # bootstrapping should globally increase the indices values within the in_base
    # will not work on unrealistic values such as a constant temperature
    assert np.count_nonzero(corrected_period > biaised_period) > np.count_nonzero(
        corrected_period < biaised_period
    )
    # bootstrapping should keep the out of base unchanged
    assert np.count_nonzero(no_bs_out_base != bs_out_base) == 0


def test_bootstrap_pr(pr_series):
    n = int(60 * 365.25)
    alpha = 0.8
    pr = pr_series(ar1(alpha, n), start="2000-01-01")
    in_base_slice = slice("2000-01-01", "2029-12-31")
    out_base_slice = slice("2030-01-01", "2059-12-31")
    per = percentile_doy(pr.sel(time=in_base_slice), per=99)

    no_bootstrap = days_over_precip_thresh(pr, per, freq="MS", bootstrap=False)
    biaised_period = no_bootstrap.sel(time=(in_base_slice))
    no_bs_out_base = no_bootstrap.sel(time=(out_base_slice))

    bootstrap = days_over_precip_thresh(pr, per, freq="MS", bootstrap=True)
    corrected_period = bootstrap.sel(time=(in_base_slice))
    bs_out_base = bootstrap.sel(time=(out_base_slice))

    # bootstrapping should globally increase the indices values within the in_base
    # will not work on unrealistic values such as a constant temperature
    assert np.count_nonzero(corrected_period > biaised_period) > np.count_nonzero(
        corrected_period < biaised_period
    )
    # bootstrapping should keep the out of base unchanged
    assert np.count_nonzero(no_bs_out_base != bs_out_base) == 0
