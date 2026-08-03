import io
import logging

from src.logging_config import configure_logging


def test_template_logger_does_not_propagate_to_root_handler():
    template_output = io.StringIO()
    root_output = io.StringIO()

    template_logger = logging.Logger("MU_PYTHON_TEMPLATE_LOGGER")
    template_logger.addHandler(logging.StreamHandler(template_output))

    root_logger = logging.Logger("test-root")
    root_logger.addHandler(logging.StreamHandler(root_output))

    configure_logging(
        template_logger,
        level_name="INFO",
        root_logger=root_logger,
    )
    template_logger.info("one record")

    assert template_output.getvalue() == "one record\n"
    assert root_output.getvalue() == ""
    assert template_logger.propagate is False


def test_configure_logging_adds_one_stdout_handler_when_root_is_unconfigured():
    output = io.StringIO()
    template_logger = logging.Logger("MU_PYTHON_TEMPLATE_LOGGER")
    root_logger = logging.Logger("test-root")

    configure_logging(
        template_logger,
        level_name="INFO",
        root_logger=root_logger,
        stream=output,
    )
    configure_logging(
        template_logger,
        level_name="INFO",
        root_logger=root_logger,
        stream=output,
    )
    root_logger.info("one root record")

    assert len(root_logger.handlers) == 1
    assert output.getvalue().count("one root record") == 1


def test_invalid_log_level_falls_back_to_warning():
    template_logger = logging.Logger("MU_PYTHON_TEMPLATE_LOGGER")
    root_logger = logging.Logger("test-root")

    configure_logging(
        template_logger,
        level_name="not-a-level",
        root_logger=root_logger,
        stream=io.StringIO(),
    )

    assert template_logger.level == logging.WARNING
    assert root_logger.level == logging.WARNING
