from typing import TypeVar, Type

from typing_extensions import Any, Optional, Union, Iterable, Dict

from . import symbolic
from .failures import MultipleSolutionFound
from .symbolic import SymbolicExpression
from .utils import render_tree

T = TypeVar('T')  # Define type variable "T"


def an(entity_var: T) -> T:
    return entity_var


def the(entity_var: T) -> T:
    first_val = next(entity_var)
    try:
        second_val = next(entity_var)
    except StopIteration:
        return first_val
    else:
        raise MultipleSolutionFound(first_val, second_val)


def entity(entity_var: T, properties: Union[SymbolicExpression, bool], show_tree: bool = False) -> Iterable[T]:
    sol_gen = evaluate_tree([entity_var], properties, show_tree=show_tree)
    for sol in sol_gen:
        yield sol[entity_var.id_].value


def set_of(entity_var: Iterable[T], properties: Union[SymbolicExpression, bool], show_tree: bool = False) -> Iterable[
    Dict[T, T]]:
    sol_gen = evaluate_tree(entity_var, properties, show_tree=show_tree)
    if isinstance(entity_var, SymbolicExpression):
        entity_var = [entity_var]
    for sol in sol_gen:
        yield {var: sol[var.id_].value for var in entity_var}


def evaluate_tree(selected_vars: Iterable[T], root: SymbolicExpression, show_tree: bool = False) -> Iterable[T]:
    if show_tree:
        render_tree(root.node_, True, view=True)
    return root.evaluate_(selected_vars)


def let(type_: Type[T], domain: Optional[Any] = None) -> T:
    return symbolic.Variable.from_domain_((v for v in domain if isinstance(v, type_)), clazz=type_)
