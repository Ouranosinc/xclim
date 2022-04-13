import logging
import sys
from pathlib import Path

from xclim import __version__
from xclim.testing._utils import ContextLogger, _logging_examples  # noqa


class TestLoggingFuncs:
    def test_logging_with_caplog_levels(self, caplog):
        with ContextLogger(caplog):
            caplog.set_level(logging.CRITICAL)

            _logging_examples()

            assert ("xclim.testing._utils", 50, "5") in caplog.record_tuples
            assert ("xclim.testing._utils", 40, "4") not in caplog.record_tuples

    def test_default_logging_setup_for_releases(self, caplog):
        _logging_examples()  # noqa

        if __version__.endswith("beta"):
            assert ("xclim.testing._utils", 40, "4") in caplog.record_tuples
        else:
            assert ("xclim.testing._utils", 40, "4") not in caplog.record_tuples

    def test_logger_configuration(self, capsys):
        with ContextLogger() as _logger:

            _logger.add(sys.stderr, level="ERROR")
            _logger.add(sys.stdout, level="INFO")

            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "WARNING" not in captured.err
            assert "WARNING" in captured.out

    def test_disabled_enabled_logger(self, capsys):
        with ContextLogger() as _logger:

            _logger.add(sys.stdout, level="INFO")
            _logger.add(sys.stderr, level="CRITICAL")

            # disable xclim logging
            _logger.disable("xclim")
            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "INFO" not in captured.out
            assert "CRITICAL" not in captured.err

            # enable xclim logging
            _logger.enable("xclim")
            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "INFO" in captured.out
            assert "WARNING" not in captured.err
            assert "CRITICAL" in captured.err

    def test_file_logger_enable(self, tmpdir):
        with ContextLogger() as _logger:

            test_log = Path(tmpdir).joinpath("xclim_test.log")
            _logger.add(test_log, level=logging.WARNING)

            _logging_examples()

            assert "INFO" not in test_log.read_text()
            assert "CRITICAL" in test_log.read_text()
