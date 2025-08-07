from typing import TypeVar, Type

from typing_extensions import Any, Optional, Union, Iterable, Dict

from . import symbolic
from .symbolic import Variable, and_, SymbolicExpression, LogicalOperator, Comparator, ConstrainingOperator, Or
from .utils import render_tree, make_tuple, make_list

T = TypeVar('T')  # Define type variable "T"


def entity(entity_var: T, properties: Union[SymbolicExpression, bool], show_tree: bool = False) -> Iterable[T]:
    if show_tree:
        render_tree(properties.node_.root, True, "query_tree", view=True, use_legend=False)
    sol_gen = properties.root_.evaluate_()
    for sol in sol_gen:
        yield sol[entity_var.id_].value


def entities(entity_var: Iterable[T], properties: Union[SymbolicExpression, bool], show_tree: bool = False) -> Iterable[Dict[T, T]]:
    if show_tree:
        render_tree(properties.node_.root, True, "query_tree", view=True, use_legend=False)
    sol_gen = properties.root_.evaluate_()
    if isinstance(entity_var, SymbolicExpression):
        entity_var = [entity_var]
    for sol in sol_gen:
        yield {var: sol[var.id_].value for var in entity_var}


def an(entity_type: Type[T], domain: Optional[Any] = None) -> Union[T, Iterable[T]]:
    return symbolic.Variable.from_domain_((v for v in domain if isinstance(v, entity_type)), clazz=entity_type)
