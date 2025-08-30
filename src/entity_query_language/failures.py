"""
Custom exception types used by entity_query_language.
"""

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
