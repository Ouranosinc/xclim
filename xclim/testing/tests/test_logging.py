import logging
import sys
from pathlib import Path

import pytest
from loguru import logger

from xclim import enable_synced_logger
from xclim.testing._utils import _logging_examples  # noqa


class TestLoggingFuncs:
    def test_logging_configuration(self, capsys):
        id1 = logger.add(sys.stdout, level="ERROR")
        id2 = logger.add(sys.stderr, level="INFO")

        logger.enable("xclim")
        _logging_examples()  # noqa
        logger.disable("xclim")

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "WARNING" not in captured.out
        assert "WARNING" in captured.err

        # teardown
        logger.remove(id1)
        logger.remove(id2)

    def test_disabled_enabled_logging(self, capsys):
        id1 = logger.add(sys.stderr, level="WARNING")
        id2 = logger.add(sys.stdout, level="CRITICAL")

        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "CRITICAL" not in captured.out

        # enable xclim logging
        logger.enable("xclim")
        _logging_examples()  # noqa
        logger.disable("xclim")

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "WARNING" not in captured.out
        assert "CRITICAL" in captured.out

        # teardown
        logger.remove(id1)
        logger.remove(id2)

    @pytest.xfail("This test needs love.")
    def test_standard_logging_configuration(self, caplog):
        logger.enable("xclim")
        enable_synced_logger()

        _logging_examples()  # noqa

        assert ("xclim.testing._utils", 40, "4") in caplog.record_tuples

    def test_file_logger_enable(self, tmpdir):
        test_log = Path(tmpdir).joinpath("xclim_test.log")
        logger.add(test_log, level=logging.WARNING)

        logger.enable("xclim")
        _logging_examples()
        logger.disable("xclim")

        assert "INFO" not in test_log.read_text()
        assert "CRITICAL" in test_log.read_text()
