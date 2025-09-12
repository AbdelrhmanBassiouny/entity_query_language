from __future__ import annotations

from collections import UserDict, defaultdict
from copy import copy

from . import logger
from .enums import EQLMode, PredicateType

"""
Core symbolic expression system used to build and evaluate entity queries.

This module defines the symbolic types (variables, sources, logical and
comparison operators) and the evaluation mechanics.
"""
import contextvars
import operator
import typing
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import lru_cache

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, ClassVar, Union, Generic, TypeVar, TYPE_CHECKING
from typing_extensions import List, Tuple, Callable

from .cache_data import cache_enter_count, cache_search_count, cache_match_count, is_caching_enabled, SeenSet, \
    IndexedCache
from .failures import MultipleSolutionFound, NoSolutionFound
from .utils import IDGenerator, is_iterable, render_tree, generate_combinations
from .hashed_data import HashedValue, HashedIterable

if TYPE_CHECKING:
    from .conclusion import Conclusion

_symbolic_mode = contextvars.ContextVar("symbolic_mode", default=None)


def _set_symbolic_mode(mode: EQLMode):
    """
    Set symbolic construction mode.

    :param mode: Can be Query or Rule.
    """
    _symbolic_mode.set(mode)


def in_symbolic_mode(mode: Optional[EQLMode] = None) -> bool:
    """
    Check whether symbolic construction mode is currently active.

    :returns: True if symbolic mode is enabled, otherwise False.
    """
    current_mode = _symbolic_mode.get()
    return current_mode == mode if mode else current_mode is not None


T = TypeVar("T")

id_generator = IDGenerator()


@dataclass(eq=False)
class SymbolicExpression(Generic[T], ABC):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a tree and are evaluated lazily to produce
    bindings for variables, subject to logical constraints.

    :ivar _child_: Optional child expression.
    :ivar _id_: Unique identifier of this node.
    :ivar _node_: Backing anytree.Node for visualization and traversal.
    :ivar _conclusion_: Set of conclusion actions attached to this node.
    :ivar _yield_when_false__: If True, may yield even when the expression is false.
    :ivar _is_false_: Internal flag indicating evaluation result for this node.
    """
    _child_: Optional[SymbolicExpression] = field(init=False)
    _id_: int = field(init=False, repr=False, default=None)
    _node_: Node = field(init=False, default=None, repr=False)
    _id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}
    _conclusion_: typing.Set[Conclusion] = field(init=False, default_factory=set)
    _symbolic_expression_stack_: ClassVar[List[SymbolicExpression]] = []
    _yield_when_false__: bool = field(init=False, repr=False, default=False)
    _is_false_: bool = field(init=False, repr=False, default=False)
    _seen_parent_values_: Dict[bool, SeenSet] = field(default_factory=lambda: {True: SeenSet(), False: SeenSet()},
                                                      init=False)

    def __post_init__(self):
        self._id_ = id_generator(self)
        node_name = self._name_ + f"_{self._id_}"
        self._create_node_(node_name)
        if hasattr(self, "_child_") and self._child_ is not None:
            self._update_child_()
        if self._id_ not in self._id_expression_map_:
            self._id_expression_map_[self._id_] = self

    def _update_child_(self, child: Optional[SymbolicExpression] = None):
        child = child or self._child_
        self._child_ = self._update_children_(child)[0]

    def _update_children_(self, *children: SymbolicExpression) -> Tuple[SymbolicExpression, ...]:
        children: Dict[int, SymbolicExpression] = dict(enumerate(children))
        for k, v in children.items():
            if not isinstance(v, SymbolicExpression):
                children[k] = Variable._from_domain_([v])
        for k, v in children.items():
            if v._node_.parent is not None and isinstance(v,
                                                          (HasDomain, Source, ResultQuantifier, QueryObjectDescriptor)):
                children[k] = self._copy_child_expression_(v)
            children[k]._node_.parent = self._node_
        return tuple(children.values())

    def _copy_child_expression_(self, child: Optional[SymbolicExpression] = None) -> SymbolicExpression:
        if child is None:
            child = self._child_
        child_cp = child.__new__(child.__class__)
        child_cp.__dict__.update(child.__dict__)
        child_cp._seen_parent_values_ = {True: SeenSet(), False: SeenSet()}
        child_cp._create_node_(child._node_.name + f"_{self._id_}")
        if hasattr(child_cp, "_child_") and child_cp._child_ is not None:
            child_cp._update_child_()
        return child_cp

    def _create_node_(self, name: str):
        self._node_ = Node(name)
        self._node_._expression_ = self

    def _reset_cache_(self) -> None:
        """
        Reset the cache of the symbolic expression.
        This method should be implemented by subclasses.
        """
        self._seen_parent_values_ = {True: SeenSet(), False: SeenSet()}

    @property
    def _yield_when_false_(self) -> bool:
        return self._yield_when_false__

    @_yield_when_false_.setter
    def _yield_when_false_(self, value):
        self._yield_when_false__ = value
        for child in self._children_:
            child._yield_when_false_ = value

    @staticmethod
    def _get_var_id_(operand: SymbolicExpression) -> int:
        """
        Get the id of the variable/main-variable instance of the given operand.

        :param operand: The operand to get the variable id from.
        :type operand: SymbolicExpression
        :return: The variable id.
        """
        return SymbolicExpression._get_var_(operand)._id_

    @staticmethod
    def _get_var_(operand: SymbolicExpression) -> HasDomain:
        """
        Get the main variable instance of the given operand, if it is A/An/The then get the selected variable,
        elif it is already a variable return it otherwise raise an error.

        :param operand: The operand to get the variable from.
        :type operand: SymbolicExpression
        :return: The variable instance.
        """
        if isinstance(operand, (ResultQuantifier, Entity)):
            return operand._parent_variable_
        elif isinstance(operand, (HasDomain, Source)):
            return operand
        elif isinstance(operand, SetOf):
            raise ValueError(f"The operand {operand} does not have a single variable/HasDomain.")
        else:
            raise NotImplementedError(f"Getting the variable from operand {operand} is not handled")

    @abstractmethod
    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    def _add_conclusion_(self, conclusion: Conclusion):
        self._conclusion_.add(conclusion)

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        return HashedIterable()

    @property
    def _conclusions_of_all_descendants_(self) -> List[Conclusion]:
        return [conc for child in self._descendants_ for conc in child._conclusion_]

    @property
    def _parent_(self) -> Optional[SymbolicExpression]:
        if self._node_.parent is not None:
            return self._node_.parent._expression_
        return None

    @_parent_.setter
    def _parent_(self, value: Optional[SymbolicExpression]):
        self._node_.parent = value._node_ if value is not None else None
        if value is not None and hasattr(value, '_child_') and value._child_ is not None:
            value._child_ = self

    def _render_tree_(self, use_dot: bool = True, show_in_console: bool = False):
        render_tree(self._root_._node_, use_dot, view=use_dot, show_in_console=show_in_console)

    @property
    def _conditions_root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree that contains conditions.
        """
        conditions_root = self._root_
        while conditions_root._child_ is not None:
            conditions_root = conditions_root._child_
            if isinstance(conditions_root._parent_, Entity):
                break
        return conditions_root

    @property
    def _root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree.
        """
        return self._node_.root._expression_

    @property
    @lru_cache(maxsize=None)
    def _sources_(self) -> List[Source]:
        sources = HashedIterable()
        for variable in self._unique_variables_:
            for source in variable.value._domain_sources_:
                for leaf in source._node_.leaves:
                    sources.add(leaf._expression_)
        return [v.value for v in sources]

    @property
    @abstractmethod
    def _name_(self) -> str:
        pass

    @property
    def _all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self._descendants_

    @property
    def _all_node_names_(self) -> List[str]:
        return [node._node_.name for node in self._all_nodes_]

    @property
    def _descendants_(self) -> List[SymbolicExpression]:
        return [d._expression_ for d in self._node_.descendants]

    @property
    def _children_(self) -> List[SymbolicExpression]:
        return [c._expression_ for c in self._node_.children]

    @classmethod
    def _current_parent_(cls) -> Optional[SymbolicExpression]:
        if cls._symbolic_expression_stack_:
            return cls._symbolic_expression_stack_[-1]
        return None

    @property
    @lru_cache(maxsize=None)
    def _parent_variable_(self) -> Variable:
        return self._all_variable_instances_[0] if self._all_variable_instances_ else None

    @property
    @lru_cache(maxsize=None)
    def _unique_variables_(self) -> HashedIterable[Variable]:
        unique_variables = HashedIterable()
        for var in self._all_variable_instances_:
            unique_variables.add(var)
        return unique_variables

    @property
    @abstractmethod
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        ...

    def _is_duplicate_output_(self, output: Dict[int, HashedValue]) -> bool:
        required_vars = self._parent_._required_variables_from_child_(self, when_true=not self._is_false_)
        if not required_vars:
            return False
        required_output = {k: v for k, v in output.items() if k in required_vars}
        if self._seen_parent_values_[not self._is_false_].check(required_output):
            return True
        else:
            self._seen_parent_values_[not self._is_false_].add(required_output)
            return False

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return OR(self, other)

    def __invert__(self):
        return Not(self)

    def __enter__(self):
        node = self
        if isinstance(node, (ResultQuantifier, QueryObjectDescriptor)):
            node = node._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(node)
        return self

    def __exit__(self, *args):
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class Source(SymbolicExpression[T]):
    """
    Leaf expression that wraps a concrete Python iterable or value as a domain
    source for variables.

    :ivar _name__: Source name used in visualization and in id construction.
    :ivar _value_: Backing value or iterable.
    """
    _name__: str
    _value_: T
    _child_: Optional[SymbolicExpression[T]] = field(init=False)

    def __post_init__(self):
        if type(self) is Source:
            self._child_ = None
        super().__post_init__()

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return []

    @property
    def _name_(self) -> str:
        return self._name__

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> T:
        return self._value_

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"{self._name_} has no attribute {name}")
        return SourceAttribute(name, getattr(self._value_, name), _child_=self)

    def __call__(self, *args, **kwargs):
        return SourceCall(self._name_, self._value_(*args, **kwargs), self, args, kwargs)


@dataclass(eq=False)
class SourceAttribute(Source):
    """
    Attribute projection on a Source value.

    Created when accessing an attribute of a Source's underlying value.
    """
    _child_: Source = field(kw_only=True)

    @property
    def _name_(self) -> str:
        return f"{self._child_._name_}.{self._name__}"


@dataclass(eq=False)
class SourceCall(Source):
    """
    Call result of invoking a callable Source value.

    Represents ``source(args, kwargs)`` in the expression tree and carries
    the call arguments for naming/visualization.
    """
    _child_: Source
    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs_: Dict[str, Any] = field(default_factory=dict)

    @property
    def _name_(self) -> str:
        return (f"{self._child_._name_}({', '.join(map(str, self._args_))},"
                f" {', '.join(f'{k}={v}' for k, v in self._kwargs_.items())})")


@dataclass(eq=False)
class CanBehaveLikeAVariable(SymbolicExpression[T], ABC):
    """
    This class adds the monitoring/tracking behaviour on variables that tracks attribute access, calling,
    and comparison operations.
    """
    _var_: CanBehaveLikeAVariable[T] = field(init=False, default=None)
    """
    A variable that is used if the child class to this class want to provide a variable to be tracked other than 
    itself, this is specially useful for child classes that holds a variable instead of being a variable and want
     to delegate the variable behaviour to the variable it has instead.
    For example, this is the case for the ResultQuantifiers & QueryDescriptors that operate on a single selected
    variable.
    """

    def __getattr__(self, name: str) -> CanBehaveLikeAVariable[T]:
        if not self._var_:
            return super().__getattr__(name)
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"{self._var_._name_} has no attribute {name}")
        return Attribute(self._var_, name)

    def __call__(self, *args, **kwargs) -> CanBehaveLikeAVariable[T]:
        if not self._var_:
            return super().__call__(*args, **kwargs)
        else:
            return Call(self._var_, args, kwargs)

    def __eq__(self, other) -> Comparator:
        if not self._var_:
            return super().__eq__(other)
        return Comparator(self._var_, other, operator.eq)

    def __contains__(self, item) -> Comparator:
        if not self._var_:
            return super().__contains__(item)
        return Comparator(item, self._var_, operator.contains)

    def __ne__(self, other) -> Comparator:
        if not self._var_:
            return super().__ne__(other)
        return Comparator(self._var_, other, operator.ne)

    def __lt__(self, other) -> Comparator:
        if not self._var_:
            return super().__lt__(other)
        return Comparator(self._var_, other, operator.lt)

    def __le__(self, other) -> Comparator:
        if not self._var_:
            return super().__le__(other)
        return Comparator(self._var_, other, operator.le)

    def __gt__(self, other) -> Comparator:
        if not self._var_:
            return super().__gt__(other)
        return Comparator(self._var_, other, operator.gt)

    def __ge__(self, other) -> Comparator:
        if not self._var_:
            return super().__ge__(other)
        return Comparator(self._var_, other, operator.ge)


@dataclass(eq=False)
class ResultQuantifier(CanBehaveLikeAVariable[T], ABC):
    """
    Base for quantifiers that return concrete results from entity/set queries
    (e.g., An, The).
    """
    _child_: QueryObjectDescriptor[T]
    _original_id_: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._var_ = self._child_._var_
        self._original_id_ = self._id_
        if isinstance(self._child_, Entity):
            self._id_ = self._child_.selected_variable._id_

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def evaluate(self) -> Union[Iterable[T], T, Iterable[Dict[SymbolicExpression[T], T]]]:
        """
        This is the method called by the user to evaluate the full query.
        """
        ...

    def _reset_cache_(self) -> None:
        super()._reset_cache_()
        self._child_._reset_cache_()

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        if self._parent_:
            vars = self._parent_._required_variables_from_child_(self, when_true=when_true)
        else:
            vars = HashedIterable()
        if isinstance(self._child_, Entity):
            vars.add(self._child_.selected_variable)
            vars.update(self._child_.selected_variable._unique_variables_)
        elif isinstance(self._child_, SetOf):
            for var in self._child_.selected_variables:
                vars.add(var)
                vars.update(var._unique_variables_)
        return vars

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_

    def _process_result_(self, result: Dict[int, HashedValue]) -> Union[T, UnificationDict]:
        if isinstance(self._child_, Entity):
            return result[self._child_.selected_variable._id_].value
        elif isinstance(self._child_, SetOf):
            selected_variables_ids = [v._id_ for v in self._child_.selected_variables]
            return UnificationDict({self._id_expression_map_[var_id]: value
                                    for var_id, value in result.items() if var_id in selected_variables_ids})
        else:
            raise NotImplementedError(f"Unknown child type {type(self._child_)}")


class UnificationDict(UserDict):
    """
    A dictionary which maps all expressions that are on a single variable to the original variable id.
    """

    def __getitem__(self, key: SymbolicExpression[T]) -> T:
        key = key._id_expression_map_[key._id_]
        return super().__getitem__(key).value


@dataclass(eq=False)
class The(ResultQuantifier[T]):
    """
    Quantifier that expects exactly one result; raises MultipleSolutionFound if more.
    """

    @property
    def _yield_when_false_(self):
        return self._yield_when_false__

    @_yield_when_false_.setter
    def _yield_when_false_(self, value):
        self._yield_when_false__ = value

    def evaluate(self) -> Union[Iterable[T], T, UnificationDict]:
        result = self._evaluate__()
        result = self._process_result_(result)
        self._reset_cache_()
        return result

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Dict[int, HashedValue]:
        sources = sources or {}
        sol_gen = self._child_._evaluate__(sources)
        result = None
        for sol in sol_gen:
            if result is None:
                result = sol
                result.update(sources)
            else:
                raise MultipleSolutionFound(result, sol)
        if result is None:
            self._is_false_ = True
        if self._is_false_:
            if self._yield_when_false_:
                return sources
            else:
                raise NoSolutionFound(self._child_)
        else:
            return result


@dataclass(eq=False)
class An(ResultQuantifier[T]):
    """Quantifier that yields all matching results one by one."""

    def evaluate(self) -> Iterable[Union[T, Dict[Union[T, SymbolicExpression[T]], T]]]:
        results = self._evaluate__()
        yield from map(self._process_result_, results)
        self._reset_cache_()

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[T]:
        sources = sources or {}
        values = self._child_._evaluate__(sources)
        for value in values:
            self._is_false_ = self._child_._is_false_
            if self._yield_when_false_ or not self._is_false_:
                value.update(sources)
                value.update({self._id_: value[self._parent_variable_._id_]})
                yield value


@dataclass(eq=False)
class QueryObjectDescriptor(CanBehaveLikeAVariable[T], ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """
    selected_variables: List[SymbolicExpression[T]] = field(default_factory=list)
    _child_: Optional[SymbolicExpression[T]] = field(default=None)
    warned_vars: typing.Set = field(default_factory=set, init=False)
    rule_mode: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        if in_symbolic_mode(EQLMode.Rule):
            self.rule_mode = True

    def _evaluate_(self, selected_vars: Optional[Iterable[HasDomain]] = None,
                   sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        self._inform_selected_variables_that_they_should_be_inferred_()
        sources = sources or {}
        if self._child_:
            child_values = self._child_._evaluate__(sources)
        else:
            child_values = [{}]
        for v in child_values:
            v.update(sources)
            if self._child_:
                self._is_false_ = self._child_._is_false_
            if self._is_false_ and not self._yield_when_false_:
                continue
            if self._child_:
                for conclusion in self._child_._conclusion_:
                    v = conclusion._evaluate__(v)
            self._warn_on_unbound_variables_(v, selected_vars)
            if selected_vars:
                var_val_gen = {var: var._evaluate__(v) for var in selected_vars}
                original_v = v
                for sol in generate_combinations(var_val_gen):
                    v = copy(original_v)
                    var_val = {var._id_: sol[var][var._id_] for var in selected_vars}
                    v.update(var_val)
                    if not self._is_duplicate_output_(v):
                        yield v
            else:
                if not self._is_duplicate_output_(v):
                    yield v

    def _reset_cache_(self) -> None:
        super()._reset_cache_()
        if self._child_:
            self._child_._reset_cache_()

    def _warn_on_unbound_variables_(self, sources: Dict[int, HashedValue], selected_vars: Iterable[HasDomain]):
        """
        Warn the user if there are unbound variables in the query descriptor, because this will result in a cartesian
        product join operation.

        :param sources: The bound values after applying the conditions.
        :param selected_vars: The variables selected in the query descriptor.
        """
        unbound_variables = HashedIterable()
        for var in selected_vars:
            unbound_variables.update(var._unique_variables_.difference(HashedIterable(values=sources)))
        unbound_variables_with_domain = HashedIterable()
        for var in unbound_variables:
            if var.value._domain_ and len(var.value._domain_.values) > 20:
                if var not in self.warned_vars:
                    self.warned_vars.add(var)
                    unbound_variables_with_domain.add(var)
        if unbound_variables_with_domain:
            logger.warning(f"\nCartesian Product: "
                           f"The following variables are not constrained "
                           f"{unbound_variables_with_domain.unwrapped_values}"
                           f"\nfor the query descriptor {self._name_}")

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        if self._child_:
            return self._child_._all_variable_instances_
        else:
            return []

    def _inform_selected_variables_that_they_should_be_inferred_(self):
        if self.rule_mode and self._child_ and self._child_ is self._conditions_root_:
            for selected_variable in self.selected_variables:
                selected_variable._is_inferred_ = True

    def __repr__(self):
        return self._name_


@dataclass(eq=False)
class SetOf(QueryObjectDescriptor[T]):
    """
    A query over a set of variables.
    """

    @property
    def _name_(self) -> str:
        return f"SetOf({', '.join(var._name_ for var in self.selected_variables)})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = self._parent_._required_variables_from_child_(self, when_true=when_true)
        required_vars.update(self.selected_variables)
        for var in self.selected_variables:
            required_vars.update(var._unique_variables_)
        if child:
            for conc in child._conclusion_:
                required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        sol_gen = self._evaluate_(self.selected_variables, sources)
        for sol in sol_gen:
            sol.update(sources)
            if self.selected_variables:
                var_val = {var._id_: next(var._evaluate__(sol))[var._id_]
                           for var in self.selected_variables if var._id_ in sol}
                sol.update(var_val)
                yield sol
            else:
                yield sol

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        vars = []
        if self.selected_variables:
            vars.extend(self.selected_variables)
        if self._child_:
            vars.extend(self._child_._all_variable_instances_)
        return vars


@dataclass(eq=False)
class Entity(QueryObjectDescriptor[T]):
    """
    A query over a single variable.
    """
    selected_variable: Optional[Variable[T]] = field(default=None, init=False)
    domain: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.selected_variable is not None:
            self.selected_variables.append(self.selected_variable)
        super().__post_init__()
        self.update_child_expression_with_variable_properties()
        self._var_ = self.selected_variable

    def update_child_expression_with_variable_properties(self):
        """
        Update the child expression with the properties of the variable.
        """
        if self.selected_variable and self.selected_variable._properties_ is not None:
            if self._child_:
                self._update_child_(chained_logic(AND, self._child_, *self.selected_variable._properties_))
            else:
                self._update_child_(chained_logic(AND, *self.selected_variable._properties_))

    @property
    def _name_(self) -> str:
        return f"Entity({self.selected_variable._name_ if self.selected_variable else ''})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = self._parent_._required_variables_from_child_(self, when_true=when_true)
        if self.selected_variable:
            required_vars.add(self.selected_variable)
            required_vars.update(self.selected_variable._unique_variables_)
        if child:
            for conc in child._conclusion_:
                required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[T]:
        selected_variables = [self.selected_variable] if self.selected_variable else []
        sol_gen = self._evaluate_(selected_variables, sources)
        for sol in sol_gen:
            sol.update(sources)
            if self._yield_when_false_ or not self._is_false_:
                if self.selected_variable:
                    for var_val in self.selected_variable._evaluate__(sol):
                        sol.update(var_val)
                        yield sol
                else:
                    yield sol

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        vars = []
        if self.selected_variable:
            vars.append(self.selected_variable)
        if self._child_:
            vars.extend(self._child_._all_variable_instances_)
        return vars


@dataclass(eq=False)
class HasDomain(CanBehaveLikeAVariable[T], ABC):
    _domain_: HashedIterable[Any] = field(default=None, init=False)
    _domain_sources_: Optional[List[Union[HasDomain, Source]]] = field(default_factory=list, init=False)
    _child_: Optional[HasDomain] = field(init=False)

    def __post_init__(self):
        self._update_domain_(self._domain_)
        super().__post_init__()

    def _update_domain_(self, domain):
        if domain is not None:
            if isinstance(domain, SymbolicExpression):
                self._domain_sources_.append(domain)
            if isinstance(domain, Source):
                domain = domain._evaluate__().value
            elif isinstance(domain, SymbolicExpression):
                domain = (next(iter(v.values())) for v in domain._evaluate__())
            elif not is_iterable(domain):
                domain = [HashedValue(domain)]
            if isinstance(self._domain_, HashedIterable):
                self._domain_.set_iterable(domain)
            else:
                self._domain_ = HashedIterable(domain)

    def __iter__(self):
        for v in self._domain_:
            yield {self._id_: HashedValue(v)}

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class Variable(HasDomain[T]):
    _name__: str
    """
    The name of the variable.
    """
    _type_: Type
    """
    The class that this variable represents.
    """
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    """
    The properties of the variable as keyword arguments.
    """
    _domain_: Union[HashedIterable, HasDomain, Source, Iterable] = field(default_factory=HashedIterable, kw_only=True)
    _invert_: bool = field(default=False)
    _predicate_type_: Optional[PredicateType] = field(default=None)
    """
    If this symbol is an instance of the Predicate class.
    """
    _is_inferred_: bool = field(default=False)
    """
    Whether this variable should be inferred.
    """
    _cache_: ClassVar[Dict[Type, IndexedCache]] = defaultdict(IndexedCache)
    """
    A mapping from variable type to an indexed cache of all seen inputs and outputs of the variable type. 
    """
    _child_vars_: Optional[Dict[str, Variable]] = field(default=None, init=False)
    """
    A dictionary mapping child variable names to variables, these are from the _kwargs_ dictionary. 
    """

    def __post_init__(self):
        self._validate_inputs_and_fill_missing_ones_()
        super().__post_init__()
        self._update_domain_sources_from_kwargs_()

    def _validate_inputs_and_fill_missing_ones_(self):
        if self._kwargs_ and not self._type_:
            raise ValueError(f"Variable {self._name_} has class keyword arguments but no type is specified.")
        if type(self) is Variable:
            self._child_ = None

    def _update_domain_sources_from_kwargs_(self):
        if self._kwargs_:
            domain_sources = []
            self._child_vars_ = {}
            for k, v in self._kwargs_.items():
                if isinstance(v, SymbolicExpression):
                    domain_sources.append(v)
                else:
                    domain_sources.append(Variable._from_domain_(v, name=self._name_ + '.' + k))
                self._child_vars_[k] = domain_sources[-1]
            self._domain_sources_.extend(domain_sources)
            self._update_children_(*domain_sources)

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        A variable either is already bound in sources by other constraints (Symbolic Expressions).,
        or has no domain and will instantiate new values by constructing the type if the type is given,
        or will yield from current domain if exists.
        """
        sources = sources or {}
        if self._id_ in sources:
            yield {self._id_: sources[self._id_]}
        elif self._domain_:
            yield from self
        elif self._type_:
            yield from self._instantiate_new_values_or_yield_from_cache_(sources)

    def _instantiate_new_values_or_yield_from_cache_(self, sources: Optional[Dict[int, HashedValue]] = None) \
            -> Iterable[Dict[int, HashedValue]]:
        if self._child_vars_:
            kwargs_generators = {k: v._evaluate__(sources)
                                 for k, v in self._child_vars_.items()}
            kwargs_combinations = generate_combinations(kwargs_generators)
        else:
            kwargs_combinations = [{}]
        for kwargs in kwargs_combinations:
            retrieved = False
            if not self._is_inferred_:
                for v in self._search_and_yield_from_cache_(**kwargs):
                    retrieved = True
                    v.update(sources)
                    yield v
            if not retrieved and (self._is_inferred_ or self._is_predicate_):
                unwrapped_hashed_kwargs = {k: v[self._get_var_id_(self._child_vars_[k])] for k, v in kwargs.items()}
                unbound_kwargs = {k: v._evaluate__(sources)
                                  for k, v in self._child_vars_.items() if k not in unwrapped_hashed_kwargs}
                for extra_kwargs in generate_combinations(unbound_kwargs):
                    kwargs.update(extra_kwargs)
                    unwrapped_hashed_kwargs.update({k: v[self._get_var_id_(self._child_vars_[k])]
                                                    for k, v in extra_kwargs.items()})
                    instance = self._type_(**{k: v.value for k, v in unwrapped_hashed_kwargs.items()})
                    yield from self._process_output_and_update_values_(instance, **kwargs)

    @property
    def _is_predicate_(self):
        return self._predicate_type_ in [PredicateType.SubClassOfPredicate, PredicateType.DecoratedMethod]

    def _search_and_yield_from_cache_(self, **kwargs):
        unwrapped_hashed_kwargs = {k: v[self._get_var_id_(self._child_vars_[k])] if isinstance(v, dict) else v
                                   for k, v in kwargs.items()}
        if self._type_ not in self._cache_.keys() or not self._cache_[self._type_].keys:
            self._cache_[self._type_].keys = list(unwrapped_hashed_kwargs.keys())
        cache_keys = [self._type_]
        if isinstance(self._type_, type):
            for t in self._cache_.keys():
                if t is self._type_:
                    continue
                if isinstance(t, type) and issubclass(t, self._type_):
                    cache_keys.append(t)
        for t in cache_keys:
            for found_kwargs, value in self._cache_[t].retrieve(unwrapped_hashed_kwargs):
                yield from self._process_output_and_update_values_(value.value, **kwargs)

    def _process_output_and_update_values_(self, function_output: Any, **kwargs) -> Iterable[Dict[int, HashedValue]]:
        """
        Process the output of the predicate/variable and get the results.

        :param function_output: The output of the predicate.
        :param kwargs: The keyword arguments of the predicate/variable.
        :return: The results' dictionary.
        """
        if self._predicate_type_ == PredicateType.SubClassOfPredicate:
            function_output = function_output()
        if (not self._invert_ and function_output) or (self._invert_ and not function_output):
            self._is_false_ = False
        else:
            self._is_false_ = True
        if self._yield_when_false_ or not self._is_false_:
            values = {self._id_: HashedValue(function_output)}
            for k, v in kwargs.items():
                values.update(v)
            yield values

    @property
    def _name_(self):
        return self._name__

    @classmethod
    def _from_domain_(cls, iterable, clazz: Optional[Type] = None,
                      name: Optional[str] = None) -> Variable:
        if not isinstance(iterable, SymbolicExpression) and not is_iterable(iterable):
            iterable = HashedIterable([iterable])
        if not clazz:
            if isinstance(iterable, (Entity, ResultQuantifier, Variable)):
                clazz = iterable._parent_variable_._type_
            else:
                clazz = type(next(iter(iterable)))
        if name is None:
            if clazz is not None:
                name = clazz.__name__
            else:
                name = "Var"
        return Variable(name, clazz, _domain_=iterable)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        variables = [self]
        if self._kwargs_:
            for v in self._kwargs_.values():
                if isinstance(v, HasDomain):
                    variables.extend(v._all_variable_instances_)
        return variables

    def __repr__(self):
        return self._name_


@dataclass(eq=False)
class DomainMapping(CanBehaveLikeAVariable[T], ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """
    _child_: CanBehaveLikeAVariable[T]
    _invert_: bool = field(init=False, default=False)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_ if not isinstance(self._child_, Variable) \
            else [self._child_]

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) \
            -> Iterable[Dict[int, HashedValue]]:
        child_val = self._child_._evaluate__(sources)
        for child_v in child_val:
            v = self._apply_mapping_(child_v[self._child_._id_])
            if (not self._invert_ and v.value) or (self._invert_ and not v.value):
                self._is_false_ = False
            else:
                self._is_false_ = True
            if self._yield_when_false_ or not self._is_false_:
                child_v[self._id_] = v
                yield child_v

    @abstractmethod
    def _apply_mapping_(self, value: HashedValue) -> HashedValue:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass


@dataclass(eq=False)
class Attribute(DomainMapping):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """
    _attr_name_: str

    def _apply_mapping_(self, value: HashedValue) -> HashedValue:
        return HashedValue(id_=value.id_, value=getattr(value.value, self._attr_name_))

    @property
    def _name_(self):
        return f"{self._child_._name_}.{self._attr_name_}"


@dataclass(eq=False)
class Call(DomainMapping):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """
    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs_: Dict[str, Any] = field(default_factory=dict)

    def _apply_mapping_(self, value: HashedValue) -> HashedValue:
        if len(self._args_) > 0 or len(self._kwargs_) > 0:
            return HashedValue(id_=value.id_, value=value.value(*self._args_, **self._kwargs_))
        else:
            return HashedValue(id_=value.id_, value=value.value())

    @property
    def _name_(self):
        return f"{self._child_._name_}()"


@dataclass(eq=False)
class BinaryOperator(SymbolicExpression, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """
    left: SymbolicExpression
    right: SymbolicExpression
    _child_: SymbolicExpression = field(init=False, default=None)
    _cache_: IndexedCache = field(default_factory=IndexedCache, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.left, self.right = self._update_children_(self.left, self.right)
        self._cache_.keys = list(self.left._unique_variables_.union(self.right._unique_variables_).values.keys())

    def yield_from_cache(self, variables_sources, cache: Optional[IndexedCache] = None):
        cache = self._cache_ if cache is None else cache
        entered = False
        for output, is_false in cache.retrieve(variables_sources):
            entered = True
            self._is_false_ = is_false
            cache_match_count.values[self._node_.name] += 1
            if is_false and self._is_duplicate_output_(output):
                continue
            yield output
        if not entered:
            cache_match_count.values[self._node_.name] += 1
        cache_enter_count.values[self._node_.name] = cache.enter_count
        cache_search_count.values[self._node_.name] = cache.search_count

    def update_cache(self, values: Dict[int, HashedValue], cache: Optional[IndexedCache] = None):
        if not is_caching_enabled():
            return
        cache = self._cache_ if cache is None else cache
        cache.insert({k: v for k, v in values.items() if k in cache.keys}, output=self._is_false_)

    def _reset_cache_(self) -> None:
        super()._reset_cache_()
        self.left._reset_cache_()
        self.right._reset_cache_()

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left._all_variable_instances_ + self.right._all_variable_instances_

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        if not child:
            child = self.left
        required_vars = HashedIterable()
        if child is self.left:
            required_vars.update(self.right._unique_variables_)
        for conc in self._conclusion_:
            required_vars.update(conc._unique_variables_)
        required_vars.update(self._parent_._required_variables_from_child_(self, when_true))
        return required_vars


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """
    left: CanBehaveLikeAVariable
    right: CanBehaveLikeAVariable
    operation: Callable[[Any, Any], bool]
    _invert__: bool = field(init=False, default=False)

    @property
    def _invert_(self):
        return self._invert__

    @_invert_.setter
    def _invert_(self, value):
        if value == self._invert__:
            return
        self._invert__ = value
        prev_operation = self.operation
        match self.operation:
            case operator.lt:
                self.operation = operator.ge if self._invert_ else self.operation
            case operator.gt:
                self.operation = operator.le if self._invert_ else self.operation
            case operator.le:
                self.operation = operator.gt if self._invert_ else self.operation
            case operator.ge:
                self.operation = operator.lt if self._invert_ else self.operation
            case operator.eq:
                self.operation = operator.ne if self._invert_ else self.operation
            case operator.ne:
                self.operation = operator.eq if self._invert_ else self.operation
            case operator.contains:
                def not_contains(a, b):
                    return not operator.contains(a, b)

                self.operation = not_contains if self._invert_ else self.operation
            case _:
                raise ValueError(f"Unsupported operation: {self.operation.__name__}")
        self._node_.name = self._node_.name.replace(prev_operation.__name__, self.operation.__name__)

    @property
    def _name_(self):
        return self.operation.__name__

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        Compares the left and right symbolic variables using the "operation".

        :param sources: Dictionary of symbolic variable id to a value of that variable, the left and right values
        will retrieve values from sources if they exist, otherwise will directly retrieve them from the original
        sources.
        :return: Yields a HashedIterable mapping a symbolic variable id to a value of that variable, it will contain
         only two values, the left and right symbolic values.
        """
        sources = sources or {}

        if is_caching_enabled():
            if self._cache_.check(sources):
                yield from self.yield_from_cache(sources)
                return

        first_operand, second_operand = self.get_first_second_operands(sources)
        first_values = first_operand._evaluate__(sources)
        for first_value in first_values:
            first_value.update(sources)
            operand_value_map = {first_operand._id_: first_value[self._get_var_id_(first_operand)]}
            second_values = second_operand._evaluate__(first_value)
            for second_value in second_values:
                operand_value_map[second_operand._id_] = second_value[self._get_var_id_(second_operand)]
                res = self.apply_operation(operand_value_map)
                self._is_false_ = not res
                if res or self._yield_when_false_:
                    values = first_value
                    values.update(second_value)
                    values.update(operand_value_map)
                    self.update_cache(values)
                    yield values

    def apply_operation(self, operand_values: Dict[int, HashedValue]):
        return self.operation(operand_values[self.left._id_].value, operand_values[self.right._id_].value)

    def get_first_second_operands(self, sources: Dict[int, HashedValue]) -> Tuple[
        SymbolicExpression, SymbolicExpression]:
        if sources and self.right._parent_variable_._id_ in sources:
            return self.right, self.left
        else:
            return self.left, self.right

    def get_result_domain(self, operand_value_map: Dict[HasDomain, HashedValue]) -> HashedIterable:
        left_leaf_value = self.left._parent_variable_._domain_[operand_value_map[self.left].id_]
        right_leaf_value = self.right._parent_variable_._domain_[operand_value_map[self.right].id_]
        return HashedIterable(values={self.left._parent_variable_._id_: left_leaf_value,
                                      self.right._parent_variable_._id_: right_leaf_value})


@dataclass(eq=False)
class LogicalOperator(BinaryOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions.
    """

    right_cache: IndexedCache = field(default_factory=IndexedCache, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.right_cache.keys = [v.id_ for v in self.right._unique_variables_]

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass(eq=False)
class AND(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    seen_left_values: SeenSet = field(default_factory=SeenSet, init=False)

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        # init an empty source if none is provided
        sources = sources or {}

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value.update(sources)
            if self._yield_when_false_ and self.left._is_false_:
                self._is_false_ = True
                if self._is_duplicate_output_(left_value):
                    continue
                yield left_value
                continue

            if is_caching_enabled() and self.right_cache.check(left_value):
                yield from self.yield_from_cache(left_value, self.right_cache)
                continue

            # constrain right values by available sources
            right_values = self.right._evaluate__(left_value)

            # For the found left value, find all right values,
            # and yield the (left, right) results found.
            for right_value in right_values:
                output = copy(right_value)
                output.update(left_value)
                self._is_false_ = self.right._is_false_
                if self._is_false_ and self._is_duplicate_output_(output):
                    continue
                self.update_cache(right_value, self.right_cache)
                yield output


@dataclass(eq=False)
class OR(LogicalOperator):
    """
    A symbolic single choice operation that can be used to choose between multiple symbolic expressions.
    """

    def __post_init__(self):
        super().__post_init__()
        self.left._yield_when_false_ = True

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None,
                                        when_true: Optional[bool] = None):
        if not child:
            child = self.left
        required_vars = HashedIterable()
        when_false = not when_true if when_true is not None else None
        if child is self.left:
            if when_false or (when_false is None):
                required_vars.update(self.right._unique_variables_)
                required_vars.update(self._parent_._required_variables_from_child_(self, None))
            else:
                required_vars.update(self._parent_._required_variables_from_child_(self, True))
            for conc in self.left._conclusion_:
                required_vars.update(conc._unique_variables_)
        elif child is self.right:
            if when_true or (when_true is None):
                for conc in self.right._conclusion_:
                    required_vars.update(conc._unique_variables_)
            required_vars.update(self._parent_._required_variables_from_child_(self, when_true))
        return required_vars

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle ElseIf logic.
        """
        # init an empty source if none is provided
        sources = sources or {}

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value.update(sources)
            if self.left._is_false_:
                if is_caching_enabled() and self.right_cache.check(left_value):
                    yield from self.yield_from_cache(left_value, self.right_cache)
                    continue
                right_values = self.right._evaluate__(left_value)
                for right_value in right_values:
                    self._is_false_ = self.right._is_false_
                    output = copy(left_value)
                    output.update(right_value)
                    if self._is_false_ and not self._yield_when_false_:
                        continue
                    if self._is_false_ and self._is_duplicate_output_(output):
                        continue
                    self.update_cache(right_value, self.right_cache)
                    yield output
            else:
                self._is_false_ = False
                yield left_value


def Not(operand: Any) -> SymbolicExpression:
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Variable._from_domain_(operand)
    if isinstance(operand, ResultQuantifier):
        raise NotImplementedError(f"Symbolic NOT operations on {ResultQuantifier} operands "
                                  f"are not allowed, you can negate the conditions or {QueryObjectDescriptor}"
                                  f" instead as negating quantifiers is most likely not what you want"
                                  f" as it is ambiguous.")
    elif isinstance(operand, Entity):
        operand = operand.__class__(Not(operand._child_), operand.selected_variable)
    elif isinstance(operand, SetOf):
        operand = operand.__class__(Not(operand._child_), operand.selected_variables)
    elif isinstance(operand, AND):
        operand = OR(Not(operand.left), Not(operand.right))
    elif isinstance(operand, OR):
        for child in operand.left._descendants_:
            child._yield_when_false_ = False
        operand.left._yield_when_false_ = False
        operand = AND(Not(operand.left), Not(operand.right))
    else:
        operand._invert_ = True
    return operand


def chained_logic(operator: Type[LogicalOperator], *conditions):
    """
    A chian of logic operation over multiple conditions, e.g. cond1 | cond2 | cond3.

    :param operator: The symbolic operator to apply between the conditions.
    :param conditions: The conditions to be chained.
    """
    prev_operation = None
    for condition in conditions:
        if prev_operation is None:
            prev_operation = condition
            continue
        prev_operation = operator(prev_operation, condition)
    return prev_operation
