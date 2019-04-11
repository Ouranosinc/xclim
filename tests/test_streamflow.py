from xclim import streamflow
import numpy as np


def test_base_flow_index(ndq_series):
    out = streamflow.base_flow_index(ndq_series, freq='YS')
    assert out.attrs['units'] == ''


class Test_FA:

    def test_simple(self, ndq_series):
        out = streamflow.freq_analysis(ndq_series, mode='max', t=[2, 5], dist='gamma', season='DJF')
        assert out.long_name == 'N-year return period max winter 1-day flow'

    def test_no_indexer(self, ndq_series):
        out = streamflow.freq_analysis(ndq_series, mode='max', t=[2, 5], dist='gamma')
        assert out.long_name == 'N-year return period max annual' \
                                ' 1-day flow'


def test_stats(ndq_series):
    out = streamflow.stats(ndq_series, freq='YS', op='min', season='MAM')
    assert out.attrs['units'] == 'm^3 s-1'


def test_qdoy_max(ndq_series, q_series):
    out = streamflow.doy_qmax(ndq_series, freq='YS', season='JJA')
    assert out.attrs['units'] == ''

    a = np.ones(450)
    a[100] = 2
    out = streamflow.doy_qmax(q_series(a), freq='YS')
    assert out[0] == 101
