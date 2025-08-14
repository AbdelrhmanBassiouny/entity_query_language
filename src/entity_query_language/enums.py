from enum import Enum


class RDREdge(Enum):
    Refinement = "except if"
    """
    Refinement edge, the edge that represents the refinement of an incorrectly fired rule.
    """
    Alternative = "else if"
    """
    Alternative edge, the edge that represents the alternative to the rule that has not fired.
    """
    Next = "also if"
    """
    Next edge, the edge that represents the next rule to be evaluated.
    """
    Then = "then"
    """
    Then edge, the edge that represents the connection to the conclusion.
    """