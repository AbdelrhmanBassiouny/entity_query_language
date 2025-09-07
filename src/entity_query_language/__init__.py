__version__ = "1.3.0"

import logging

logger = logging.Logger("eql")
logger.setLevel(logging.INFO)

from .entity import (entity, an, let, the, set_of,
                     and_, or_, not_, contains, in_,
                     symbol, predicate)
from .rule import refinement, alternative, symbolic_mode
from .conclusion import Add, Set
from .failures import MultipleSolutionFound, NoSolutionFound

