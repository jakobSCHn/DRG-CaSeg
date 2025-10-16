import logging
import sys
import colorlog


def setup_logging(level=logging.INFO):
    """
    Configures the root logger.
    """
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_format = (
        "[%(bold_black)s%(asctime)s%(reset)s]"
        "[%(reset)s%(log_color)s%(levelname)-8s%(reset)s%(reset)s] "
        "%(bold_black)s%(name)-20s - %(funcName)-18s,%(lineno)4d:%(reset)s "
        "%(log_color)s%(message)s%(reset)s"
    )

    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG":    "white",
                "INFO":     "white",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "red,bg_white"
            }
        },
        style="%"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)