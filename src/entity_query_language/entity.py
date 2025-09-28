from __future__ import annotations

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import Any, Optional, Union, Iterable, TypeVar, Type, Tuple, List

from .symbolic import (SymbolicExpression, Entity, SetOf, The, An, AND, Comparator, \
                       chained_logic, Not, CanBehaveLikeAVariable, ResultQuantifier, From, symbolic_mode,
                       Variable, Infer, _optimize_or, Flatten, Concatenate, ForAll)
from .predicate import Predicate, symbols_registry

T = TypeVar('T')  # Define type variable "T"


def an(entity_: Optional[Union[SetOf[T], Entity[T], T, Iterable[T], Type[T]]] = None,
       *properties: Union[SymbolicExpression, bool, Predicate],
       has_type: Optional[Type[T]] = None) -> Union[An[T], T, SymbolicExpression[T]]:
    """
    Select a single element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :param has_type: Optional type to constrain the selected variable.
    :type has_type: Optional[Type[T]]
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return select_one_or_select_many_or_infer(An, entity_, *properties, has_type=has_type)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(entity_: Union[SetOf[T], Entity[T], T, Iterable[T], Type[T], None],
        *properties: Union[SymbolicExpression, bool, Predicate]
        , has_type: Optional[Type[T]] = None) -> The[T]:
    """
    Select the unique element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :param has_type: Optional type to constrain the selected variable.
    :type has_type: Optional[Type[T]]
    :return: A quantifier representing "an" element.
    :rtype: The[T]
    """
    return select_one_or_select_many_or_infer(The, entity_, *properties, has_type=has_type)


def infer(entity_: Union[SetOf[T], Entity[T], T, Iterable[T], Type[T], None],
          *properties: Union[SymbolicExpression, bool, Predicate]
          , has_type: Optional[Type[T]] = None) -> Infer[T]:
    return select_one_or_select_many_or_infer(Infer, entity_, *properties, has_type=has_type)


def select_one_or_select_many_or_infer(quantifier: Union[Type[An], Type[The], Type[Infer]],
                                       entity_: Union[SetOf[T], Entity[T], Type[T], None],
                                       *properties: Union[SymbolicExpression, bool, Predicate],
                                       has_type: Optional[Type[T]] = None) -> Union[An[T], The[T], Infer[T]]:
    if isinstance(entity_, type):
        entity_ = entity_()
    elif entity_ is None and has_type:
        entity_ = has_type()
    if isinstance(entity_, (Entity, SetOf)):
        q = quantifier(entity_)
    elif isinstance(entity_, ResultQuantifier) and not properties:
        q = entity_
    elif isinstance(entity_, CanBehaveLikeAVariable):
        q = quantifier(entity(entity_, *properties))
    elif isinstance(entity_, (list, tuple)):
        q = quantifier(set_of(entity_, *properties))
    else:
        raise ValueError(f'Invalid entity: {entity_}')
    return q


def entity(selected_variable: T, *properties: Union[SymbolicExpression, bool, Predicate, Any]) -> Entity[T]:
    """
    Create an entity descriptor from a selected variable and its properties.

    :param selected_variable: The variable to select in the result.
    :type selected_variable: T
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :return: Entity descriptor.
    :rtype: Entity[T]
    """
    selected_variables, expression = _extract_variables_and_expression([selected_variable], *properties)
    return Entity(selected_variables=selected_variables, _child_=expression)


def set_of(selected_variables: Iterable[T], *properties: Union[SymbolicExpression, bool, Predicate]) -> SetOf[T]:
    """
    Create a set descriptor from selected variables and their properties.

    :param selected_variables: Iterable of variables to select in the result set.
    :type selected_variables: Iterable[T]
    :param properties: Conditions that define the set.
    :type properties: Union[SymbolicExpression, bool]
    :return: Set descriptor.
    :rtype: SetOf[T]
    """
    selected_variables, expression = _extract_variables_and_expression(selected_variables, *properties)
    return SetOf(selected_variables=selected_variables, _child_=expression)


def _extract_variables_and_expression(selected_variables: Iterable[T], *properties: Union[SymbolicExpression, bool]) \
        -> Tuple[List[T], SymbolicExpression]:
    """
    Extracts the variables and expressions from the selected variables, this is usefule when
    the selected variables are not all variables but some are expressions like A/An/The.

    :param selected_variables: Iterable of variables to select in the result set.
    :type selected_variables: Iterable[T]
    :param properties: Conditions on the selected variables.
    :type properties: Union[SymbolicExpression, bool]
    :return: Tuple of selected variables and expressions.
    :rtype: Tuple[List[T], List[SymbolicExpression]]
    """
    final_expression_list = list(properties)
    expression_list = []
    selected_variables = list(selected_variables)
    for i, var in enumerate(selected_variables):
        if isinstance(var, ResultQuantifier):
            result_quantifier = var
            var = var._var_
            expression_list.append(result_quantifier)
            # if result_quantifier._child_._child_:
            #     expression_list.append(result_quantifier._child_._child_)
            selected_variables[i] = var
    expression_list += final_expression_list
    expression = None
    if len(expression_list) > 0:
        expression = and_(*expression_list) if len(expression_list) > 1 else expression_list[0]
    return selected_variables, expression


def let(type_: Type[T], domain: Optional[Any] = None, name: Optional[str] = None) \
        -> Union[T, CanBehaveLikeAVariable[T], Variable[T]]:
    """
    Declare a symbolic variable or source.

    If a domain is provided, the variable will iterate over that domain; otherwise
    a free variable is returned that can be bound by constraints.

    :param type_: The expected Python type of items in the domain.
    :type type_: Type[T]
    :param domain: A value or a set of values to constrain the variable to.
    :type domain: Optional[Any]
    :param name: Variable or source name.
    :type name: str
    :return: A Variable with the given type, name, and domain.
    :rtype: T
    :raises ValueError: If the type is not registered as a symbol.
    """
    if not any(issubclass(type_, t) for t in symbols_registry):
        raise ValueError(f'Type {type_} is not registered as symbol, did you forget to decorate it with @symbol?')
    with symbolic_mode():
        if domain is None:
            var = type_()
        else:
            var = type_(From(domain))
    if name is not None:
        var._name__ = name
    return var


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
    return chained_logic(_optimize_or, *conditions)


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


def flatten(var: Union[CanBehaveLikeAVariable[T], Iterable[T]]) -> Union[CanBehaveLikeAVariable[T], Iterable[T]]:
    """
    Flatten a nested iterable domain into individual items while preserving the parent bindings.
    This returns a DomainMapping that, when evaluated, yields one solution per inner element
    (similar to SQL UNNEST), keeping existing variable bindings intact.
    """
    return Flatten(var)


def concatenate(var: Union[CanBehaveLikeAVariable[T], Iterable[T]]) -> Union[CanBehaveLikeAVariable[T], Iterable[T]]:
    """
    Concatenate a nested iterable domain into a one-element domain that is still a nested iterable that contains all
    the values of the sub iterables.
    """
    return Concatenate(var)


def for_all(universal_variable: Union[CanBehaveLikeAVariable[T], T],
            condition: Union[SymbolicExpression, bool, Predicate]):
    """
    A universal on variable that finds all sets of variable bindings (values) that satisfy the condition for every
     value of the universal_variable.

    :param universal_variable: The universal on variable that the condition must satisfy for all its values.
    :condition: A SymbolicExpression or bool representing a condition that must be satisfied.
    :return: A SymbolicExpression that can be evaluated producing every set that satisfies the condition.
    """
    return ForAll(universal_variable, condition)
