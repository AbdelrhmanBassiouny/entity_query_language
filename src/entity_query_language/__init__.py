__version__ = "0.2.1"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import entity, an, let, the, set_of
from .symbolic import symbol, And, Or, Not, contains, in_, SymbolicMode
from .failures import MultipleSolutionFound

