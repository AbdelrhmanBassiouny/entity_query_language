__version__ = "3.1.1"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import (entity, a, an, let, the, set_of,
                     and_, or_, not_, contains, in_, infer, flatten, concatenate, for_all)
from .rule import refinement, alternative
from .symbolic import symbolic_mode, From, rule_mode
from .predicate import predicate, symbol, Predicate, HasType
from .conclusion import Add, Set
from .failures import MultipleSolutionFound, NoSolutionFound

