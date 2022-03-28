import logging
import sys
from pathlib import Path

from loguru import logger

from xclim.testing._utils import _logging_examples  # noqa


class TestLoggingFuncs:
    def test_logging_configuration(self, capsys):
        id1 = logger.add(sys.stderr, level="ERROR")
        id2 = logger.add(sys.stdout, level="INFO")

        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "WARNING" not in captured.err
        assert "WARNING" in captured.out

        # teardown
        logger.remove(id1)
        logger.remove(id2)

    def test_disabled_enabled_logging(self, capsys):
        id1 = logger.add(sys.stdout, level="INFO")
        id2 = logger.add(sys.stderr, level="CRITICAL")

        logger.disable("xclim")
        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "INFO" not in captured.out
        assert "CRITICAL" not in captured.err

        # enable xclim logging
        logger.enable("xclim")
        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "INFO" in captured.out
        assert "WARNING" not in captured.err
        assert "CRITICAL" in captured.err

        # teardown
        logger.remove(id1)
        logger.remove(id2)

    def test_standard_logging_configuration(self, caplog):
        _logging_examples()  # noqa

        assert ("xclim.testing._utils", 40, "4") in caplog.record_tuples

    def test_file_logger_enable(self, tmpdir):
        test_log = Path(tmpdir).joinpath("xclim_test.log")
        id1 = logger.add(test_log, level=logging.WARNING)

        _logging_examples()

        assert "INFO" not in test_log.read_text()
        assert "CRITICAL" in test_log.read_text()

        logger.remove(id1)
