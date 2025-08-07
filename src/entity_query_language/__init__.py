__version__ = "0.0.5"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import entity, an, let
from .symbolic import symbol

