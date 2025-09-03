from __future__ import annotations
"""
Custom exception types used by entity_query_language.
"""
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from entity_query_language.symbolic import SymbolicExpression


class MultipleSolutionFound(Exception):
    """
    Raised when a query unexpectedly yields more than one solution where a single
    result was expected.

    :param first_val: The first solution encountered.
    :param second_val: The second solution encountered.
    """
    def __init__(self, first_val, second_val):
        super(MultipleSolutionFound, self).__init__(
            f"Multiple solutions found, the first two are {first_val}\n{second_val}"
        )


class NoSolutionFound(Exception):
    """
    Raised when a query does not yield any solution.
    """
    def __init__(self, expression: SymbolicExpression):
        super(NoSolutionFound, self).__init__(f"No solution found for expression {expression}")
