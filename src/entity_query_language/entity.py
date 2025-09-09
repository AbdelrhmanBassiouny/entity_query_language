from __future__ import annotations

from functools import wraps
from typing import Tuple, List

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import Any, Optional, Union, Iterable, TypeVar, Type, dataclass_transform, Callable

from .symbolic import SymbolicExpression, Entity, SetOf, The, An, Variable, AND, Comparator, \
    chained_logic, HasDomain, Source, HasType, OR, in_symbolic_mode, Not
from .predicate import SymbolicPredicate, Predicate
from .utils import is_iterable

T = TypeVar('T')  # Define type variable "T"


def an(entity_: Union[SetOf[T], Entity[T], T, Iterable[T]], *properties: Union[SymbolicExpression, bool],
       domain: Optional[Any]=None) -> Union[An[T], T, SymbolicExpression[T]]:
    """
    Select a single element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :param domain: Optional domain to constrain the selected variable.
    :type domain: Optional[Any]
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return _an_or_the(An, entity_, *properties, domain=domain)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(entity_: Union[SetOf[T], Entity[T], T, Iterable[T]], *properties: Union[SymbolicExpression, bool],
       domain: Optional[Any]=None) -> The[T]:
    """
    Select the unique element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :type entity_: Union[SetOf[T], Entity[T]]
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :param domain: Optional domain to constrain the selected variable.
    :type domain: Optional[Any]
    :return: A quantifier representing "an" element.
    :rtype: The[T]
    """
    return _an_or_the(The, entity_, *properties, domain=domain)


def _an_or_the(quantifier: Union[Type[An], Type[The]],
               entity_: Union[SetOf[T], Entity[T]], *properties: Union[SymbolicExpression, bool],
               domain: Optional[Any]=None) -> Union[An[T], The[T]]:
    if isinstance(entity_, (Entity, SetOf)):
        return quantifier(entity_)
    elif isinstance(entity_, HasDomain):
        return quantifier(entity(entity_, *properties, domain=domain))
    elif isinstance(entity_, (list, tuple)):
        return quantifier(set_of(entity_, *properties))
    else:
        raise ValueError(f'Invalid entity: {entity_}')


def entity(selected_variable: T, *properties: Union[SymbolicExpression, bool, Predicate],
           domain: Optional[Any] = None) -> Entity[T]:
    """
    Create an entity descriptor from a selected variable and its properties.

    :param selected_variable: The variable to select in the result.
    :type selected_variable: T
    :param properties: Conditions that define the entity.
    :type properties: Union[SymbolicExpression, bool]
    :param domain: Optional domain to constrain the selected variable.
    :type domain: Optional[Any]
    :return: Entity descriptor.
    :rtype: Entity[T]
    """
    selected_variables, expression = _extract_variables_and_expression([selected_variable], *properties)
    return Entity(_child_=expression, selected_variable=selected_variables[0], domain=domain)


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
    selected_variables, expression = _extract_variables_and_expression(selected_variables, *properties)
    return SetOf(_child_=expression, selected_variables=selected_variables)


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
    expression_list = list(properties)
    selected_variables = list(selected_variables)
    for i, var in enumerate(selected_variables):
        if not isinstance(var, HasDomain):
            expression = var
            var = var._get_var_(var)
            if var._properties_ and var._domain_:
                expression_list.append(expression)
            selected_variables[i] = var
    expression = None
    if len(expression_list) > 0:
        expression = and_(*expression_list) if len(expression_list) > 1 else expression_list[0]
    return selected_variables, expression


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
        return Variable(name, type_, _domain_=HasType(_child_=domain, _type_=(type_,)))
    elif is_iterable(domain):
        domain = HasType(_child_=Source(type_.__name__, domain), _type_=(type_,))
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
