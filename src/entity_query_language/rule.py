from __future__ import annotations

from contextlib import contextmanager
from typing import Union, Optional

from .entity import T
from .enums import RDREdge, EQLMode
from .symbolic import SymbolicExpression, chained_logic, AND, BinaryOperator, _set_symbolic_mode
from .conclusion_selector import ExceptIf, ElseIf


@contextmanager
def rule_mode(query: Optional[SymbolicExpression] = None):
    """
    Wrapper around symbolic construction mode to easily enable rule mode
    """
    # delegate to symbolic_mode
    with symbolic_mode(query, EQLMode.Rule) as ctx:
        yield ctx


@contextmanager
def symbolic_mode(query: Optional[SymbolicExpression] = None, mode: EQLMode = EQLMode.Query):
    """
    Context manager to temporarily enable symbolic construction mode.

    Within the context, calling classes decorated with ``@symbol`` produces
    symbolic Variables instead of real instances.

    :param query: Optional symbolic expression to also enter/exit as a context.
    """
    try:
        if query is not None:
            query.__enter__()
        _set_symbolic_mode(mode)
        yield SymbolicExpression._current_parent_()
    finally:
        if query is not None:
            query.__exit__()
        _set_symbolic_mode(None)


def refinement(*conditions: Union[SymbolicExpression[T], bool]) -> SymbolicExpression[T]:
    """
    Add a refinement branch (ExceptIf node with its right the new conditions and its left the base/parent rule/query)
     to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via ExceptIf to the current node, representing a refinement/specialization path.

    :param conditions: The refinement conditions. They are chained with AND.
    :returns: The newly created branch node for further chaining.
    """
    new_branch = chained_logic(AND, *conditions)
    prev_parent = SymbolicExpression._current_parent_()._parent_
    new_conditions_root = ExceptIf(SymbolicExpression._current_parent_(), new_branch)
    new_branch._node_.weight = RDREdge.Refinement
    new_conditions_root._parent_ = prev_parent
    return new_conditions_root.right


def alternative(*conditions: Union[SymbolicExpression[T], bool]) -> SymbolicExpression[T]:
    """
    Add an alternative branch (logical OR) to the current condition tree.

    Each provided condition is chained with AND, and the resulting branch is
    connected via OR to the current node, representing an alternative path.

    :param conditions: Conditions to chain with AND and attach as an alternative.
    :returns: The newly created branch node for further chaining.
    """
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_()
    if isinstance(current_node._parent_, ElseIf):
        current_node = current_node._parent_
    prev_parent = current_node._parent_
    new_conditions_root = ElseIf(current_node, new_branch)
    new_branch._node_.weight = RDREdge.Alternative
    new_conditions_root._parent_ = prev_parent
    if isinstance(prev_parent, BinaryOperator):
        prev_parent.right = new_conditions_root
    return new_conditions_root.right
