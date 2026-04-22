import logging

logger = logging.getLogger("Python SCPT")


class ScptError(Exception):
    """
    Error that occurs due to the nature of SCPT processing,
    e.g. input values do not inhibit the signal as expected of SCPT.
    """
