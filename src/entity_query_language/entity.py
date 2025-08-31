from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Optional, dataclass_transform, Union, Callable

from .enums import RDREdge

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import Any, Optional, Union, Iterable, TypeVar, Type

from .symbolic import SymbolicExpression, Entity, SetOf, The, An, Variable, AND, OR, Comparator, \
    chained_logic, HasDomain, Source, SourceCall, SourceAttribute, HasType, OR, _set_symbolic_mode, in_symbolic_mode, T, \
    ExceptIf, BinaryOperator, Call, Predicate, Not
from .utils import render_tree, is_iterable

T = TypeVar('T')  # Define type variable "T"


def an(entity_: Union[SetOf[T], Entity[T]]) -> An[T]:
    """
    Select a single element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return An(entity_)


def the(entity_: Union[SetOf[T], Entity[T]]) -> The[T]:
    """
    Select the unique element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :return: A quantifier representing "the" element.
    :rtype: The[T]
    """
    return The(entity_)


def entity(selected_variable: T, *properties: Union[SymbolicExpression, bool]) -> Entity[T]:
    """
    Create an entity descriptor from a selected variable and its properties.

    :param selected_variable: The variable to select in the result.
    :type selected_variable: T
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :return: Entity descriptor.
    :rtype: Entity[T]
    """
    expression = and_(*properties) if len(properties) > 1 else properties[0]
    return Entity(_child_=expression, selected_variable_=selected_variable)


def set_of(selected_variables: Iterable[T], *properties: Union[SymbolicExpression, bool]) -> SetOf[T]:
    """
    Create a set descriptor from selected variables and their properties.

    :param selected_variables: Iterable of variables to select in the result set.
    :type selected_variables: Iterable[T]
    :param properties: Conditions that define the set.
    :type properties: Union[SymbolicExpression, bool]
    :return: Set descriptor.
    :rtype: SetOf[T]
    """
    expression = and_(*properties) if len(properties) > 1 else properties[0]
    return SetOf(_child_=expression, selected_variables_=selected_variables)


def let(name: str, type_: Type[T], domain: Optional[Any] = None) -> Union[T, HasDomain, Source]:
    """
    Declare a symbolic variable or source.

    If a domain is provided, the variable will iterate over that domain; otherwise
    a free variable is returned that can be bound by constraints.

    :param name: Variable or source name.
    :type name: str
    :param type_: The expected Python type of items in the domain.
    :type type_: Type[T]
    :param domain: Either a concrete iterable domain, a HasDomain/Source, or None.
    :type domain: Optional[Any]
    :return: A Variable or a Source depending on inputs.
    :rtype: Union[T, HasDomain, Source]
    """
    if domain is None:
        return Variable(name, type_)
    elif isinstance(domain, (HasDomain, Source)):
        return Variable(name, type_, _domain_=HasType(_child_=domain, _type_=type_))
    elif is_iterable(domain):
        domain = HasType(_child_=Source(type_.__name__, domain), _type_=type_)
        return Variable(name, type_, _domain_=domain)
    else:
        return Source(name, domain)


def and_(*conditions):
    """
    Logical conjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An AND operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(AND, *conditions)


def or_(*conditions):
    """
    Logical disjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An OR operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(OR, *conditions)


def not_(operand: SymbolicExpression):
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    return Not(operand)


def contains(container, item):
    """
    Check whether a container contains an item.

    :param container: The container expression.
    :param item: The item to look for.
    :return: A comparator expression equivalent to ``item in container``.
    :rtype: SymbolicExpression
    """
    return in_(item, container)


def in_(item, container):
    """
    Build a comparator for membership: ``item in container``.

    :param item: The candidate item.
    :param container: The container expression.
    :return: Comparator expression for membership.
    :rtype: Comparator
    """
    return Comparator(container, item, operator.contains)


@contextmanager
def symbolic_mode(query: Optional[SymbolicExpression] = None):
    """
    Context manager to temporarily enable symbolic construction mode.

    Within the context, calling classes decorated with ``@symbol`` produces
    symbolic Variables instead of real instances.

    :param query: Optional symbolic expression to also enter/exit as a context.
    """
    try:
        if query is not None:
            query.__enter__()
        _set_symbolic_mode(True)
        yield SymbolicExpression._current_parent_()
    finally:
        if query is not None:
            query.__exit__()
        _set_symbolic_mode(False)



@dataclass_transform()
def symbol(cls):
    """
    Class decorator that makes a class construct symbolic Variables when inside
    a symbolic_rule context.

    When symbolic mode is active, calling the class returns a Variable bound to
    either a provided domain or to deferred keyword domain sources.

    :param cls: The class to decorate.
    :return: The same class with a patched ``__new__``.
    """
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            if len(args) > 0:
                return Variable(args[0], symbolic_cls, _domain_=args[1])
            return Variable(symbolic_cls.__name__, symbolic_cls, _cls_kwargs_=kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls


def predicate(function: Callable[..., T]) -> Callable[..., SymbolicExpression[T]]:
    """
    Function decorator that constructs a symbolic expression representing the function call
     when inside a symbolic_rule context.

    When symbolic mode is active, calling the method returns a Call instance which is a SymbolicExpression bound to
    representing the method call that is not evaluated until the evaluate() method is called on the query/rule.

    :param function: The function to decorate.
    :return: The decorated function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        if in_symbolic_mode():
            return Predicate(function, _args_=args, _kwargs_=kwargs)
        return function(*args, **kwargs)

    return wrapper


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
    if isinstance(current_node._parent_, OR):
        current_node = current_node._parent_
    prev_parent = current_node._parent_
    new_conditions_root = OR(current_node, new_branch)
    new_branch._node_.weight = RDREdge.Alternative
    new_conditions_root._parent_ = prev_parent
    if isinstance(prev_parent, BinaryOperator):
        prev_parent.right = new_conditions_root
    return new_conditions_root.right
