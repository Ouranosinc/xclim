from dataclasses import dataclass
from typing import Optional

from _pytest.mark.structures import EMPTY_PARAMETERSET_OPTION


@dataclass
class BootstrapConfig:
    percentile: int  # ]0, 100[
    in_base_slice: slice
    out_of_base_slice: slice = None  # When None, only the in-base will be computed
    percentile_window: Optional[int] = 5
    indice_window: int = None
    freq: str = "MS"


NO_BOOTSRAP: BootstrapConfig = "null"
