__version__ = "0.0.0"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import entity, an, let, the, set_of, And, Or, contains, in_
from .symbolic import symbol, SymbolicMode, Not
from .failures import MultipleSolutionFound

