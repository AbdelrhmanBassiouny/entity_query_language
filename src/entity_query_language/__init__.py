__version__ = "1.7.0"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import (entity, a, an, let, the, set_of,
                     and_, or_, not_, contains, in_)
from .predicate import predicate, symbol, Predicate
from .rule import refinement, alternative
from .symbolic import symbolic_mode, From
from .conclusion import Add, Set
from .failures import MultipleSolutionFound, NoSolutionFound

