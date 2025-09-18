from __future__ import annotations

from typing import Union

from .hashed_data import T
from .enums import RDREdge
from .symbolic import SymbolicExpression, chained_logic, AND, BinaryOperator, Variable, properties_to_expression_tree, \
    CanBehaveLikeAVariable, ResultQuantifier
from .conclusion_selector import ExceptIf, ElseIf


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
    # parsed_conditions = []
    # for condition in conditions:
    #     if isinstance(condition, Variable) and condition._child_vars_:
    #         parsed_conditions.append(properties_to_expression_tree(condition, condition._child_vars_))
    #     elif isinstance(condition, ResultQuantifier):
    #         if condition._child_._child_:
    #             parsed_conditions.append(condition._child_._child_)
    #         elif condition._var_._child_vars_:
    #             parsed_conditions.append(properties_to_expression_tree(condition._var_, condition._var_._child_vars_))
    #     else:
    #         parsed_conditions.append(condition)
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_()
    if isinstance(current_node._parent_, ElseIf):
        current_node = current_node._parent_
    prev_parent = current_node._parent_
    current_node._parent_ = None
    new_conditions_root = ElseIf(current_node, new_branch)
    new_branch._node_.weight = RDREdge.Alternative
    new_conditions_root._parent_ = prev_parent
    if isinstance(prev_parent, BinaryOperator):
        prev_parent.right = new_conditions_root
    return new_conditions_root.right
