import os
from pathlib import Path

from xclim.testing import TestDataSet

TD = Path(os.path.dirname(__file__)) / "testdata"


FWI = TestDataSet("gfwed", TD / "FWI")
FWI.add(
    "init",
    "https://portal.nccs.nasa.gov/datashare/"
    "GlobalFWI/v2.0/fwiCalcs.MERRA2/Default/MERRA2/1985/FWI.MERRA2.Daily.Default.19850730.nc",
),
FWI.add(
    "pr",
    "https://portal.nccs.nasa.gov/datashare/"
    "GlobalFWI/v2.0/wxInput/MERRA2/1985/Prec.MERRA2.Daily.Default.19850731.nc",
),
FWI.add(
    "wx",
    "https://portal.nccs.nasa.gov/datashare/"
    "GlobalFWI/v2.0/wxInput/MERRA2/1985/Wx.MERRA2.Daily.Default.19850731.nc",
),
FWI.add(
    "out",
    "https://portal.nccs.nasa.gov/datashare/"
    "GlobalFWI/v2.0/fwiCalcs.MERRA2/Default/MERRA2/1985/FWI.MERRA2.Daily.Default.19850731.nc",
)
