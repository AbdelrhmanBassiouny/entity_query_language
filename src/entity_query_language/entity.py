from __future__ import annotations
"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import Any, Optional, Union, Iterable, TypeVar, Type

from .symbolic import SymbolicExpression, Entity, SetOf, The, An, Variable, AND, OR, Comparator, \
    chained_logic, HasDomain, Source, SourceCall, SourceAttribute, HasType, OR
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
    expression = And(*properties) if len(properties) > 1 else properties[0]
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
    expression = And(*properties) if len(properties) > 1 else properties[0]
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


def And(*conditions):
    """
    Logical conjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An AND operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(AND, *conditions)


def Or(*conditions):
    """
    Logical disjunction of conditions.

    :param conditions: One or more conditions to combine.
    :type conditions: SymbolicExpression | bool
    :return: An OR operator joining the conditions.
    :rtype: SymbolicExpression
    """
    return chained_logic(OR, *conditions)


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
