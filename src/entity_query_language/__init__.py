__version__ = "0.0.2"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import entity, an
