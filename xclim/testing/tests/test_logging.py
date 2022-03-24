import sys

from loguru import logger


class TestLoggingFuncs:
    @staticmethod
    def _logging_examples() -> None:
        """Testing module"""
        logger.trace("0")
        logger.debug("1")
        logger.info("2")
        logger.success("2.5")
        logger.warning("3")
        logger.error("4")
        logger.critical("5")

    def test_logging_configuration(self, capsys):
        logger.add(sys.stderr, level="WARNING")
        logger.add(sys.stdout, level="INFO")

        self._logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "INFO" in captured.out

        # teardown
        logger.remove()

    def test_disabled_enabled_logging(self, capsys):
        logger.add(sys.stderr, level="WARNING")
        logger.add(sys.stdout, level="INFO")

        # disable xclim logging
        logger.disable("xclim")

        self._logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "INFO" not in captured.out

        # enable xclim logging
        logger.enable("xclim")

        self._logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "INFO" not in captured.err
        assert "WARNING" in captured.err
        assert "INFO" in captured.out
        logger.disable("xclim")

        # teardown
        logger.remove()
