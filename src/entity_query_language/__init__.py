__version__ = "0.1.2"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import entity, an, let, the
from .symbolic import symbol, And, Or, Not, contains, in_
from .failures import MultipleSolutionFound

