from __future__ import annotations

from typing import List

"""
Custom exception types used by entity_query_language.
"""
from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .symbolic import SymbolicExpression
    from .predicate import Predicate


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


class ValueNotFoundInCache(Exception):
    """
    Raised when a value is not found in the cache.
    """
    def __init__(self, expression: SymbolicExpression, value: Any):
        super(ValueNotFoundInCache, self).__init__(f"Value {value} not found in cache for expression {expression}")


class MoreThanOneCacheEntryMatched(Exception):
    def __init__(self, predicate: Predicate, value: Any, results: List[Any]):
        super(MoreThanOneCacheEntryMatched, self).__init__(f"More than one cache entry matched for predicate "
                                                           f"{predicate} with value {value}. Results: {results}")