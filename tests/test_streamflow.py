import pytest
from xclim import streamflow
import numpy as np


def test_base_flow_index(ndq_series):
    out = streamflow.base_flow_index(ndq_series, freq='YS')
    assert out.attrs['units'] == ''


class Test_FA():

    def test_simple(self, ndq_series):
        out = streamflow.freq_analysis(ndq_series, mode='max', t=[2, 5], dist='gamma', season='DJF')
        assert out.long_name == 'N-year return period max winter 1-day flow'

    def test_no_indexer(self, ndq_series):
        out = streamflow.freq_analysis(ndq_series, mode='max', t=[2, 5], dist='gamma')
        assert out.long_name == 'N-year return period max annual 1-day flow'


class TestStats():

    def test_simple(self, ndq_series):
        out = streamflow.stats(ndq_series, freq='YS', op='min', season='MAM')
        assert out.attrs['units'] == 'm^3 s-1'

    @pytest.mark.skip()
    def test_missing(self, ndq_series):
        a = ndq_series
        a = ndq_series.where(~((a.time.dt.dayofyear == 5) * (a.time.dt.year == 1902)))
        out = streamflow.stats(a, op='max', month=1)

        np.testing.assert_array_equal(out[1].isnull(), False)
        np.testing.assert_array_equal(out[2].isnull(), True)


def test_qdoy_max(ndq_series, q_series):
    out = streamflow.doy_qmax(ndq_series, freq='YS', season='JJA')
    assert out.attrs['units'] == ''

    a = np.ones(450)
    a[100] = 2
    out = streamflow.doy_qmax(q_series(a), freq='YS')
    assert out[0] == 101
