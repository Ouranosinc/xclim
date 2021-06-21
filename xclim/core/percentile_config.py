from dataclasses import dataclass
from typing import List, Optional, Union

from xarray.core.dataarray import DataArray


@dataclass
class PercentileConfig:
    percentile: Union[float, List[float]]
    in_base_slice: slice(str, str)
    out_of_base_slice: slice(str, str)
    # in_base_percentiles: the percentile of the in base used for to compute the exceedance of the out of base
    in_base_percentiles: DataArray = None
    # percentile_window: moving window from which each percentile is calculated
    percentile_window: Optional[int] = 5
    # indice_window: some indices are themself computed on a moving window
    indice_window: int = None
    freq: str = "MS"
