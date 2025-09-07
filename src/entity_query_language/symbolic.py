from __future__ import annotations

from copy import copy

from . import logger

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
from .utils import make_list, IDGenerator, is_iterable, render_tree, generate_combinations
from .hashed_data import HashedValue, HashedIterable

if TYPE_CHECKING:
    from .conclusion import Conclusion

_symbolic_mode = contextvars.ContextVar("symbolic_mode", default=False)


def _set_symbolic_mode(value: bool):
    """
    Set whether symbolic construction mode is active.

    :param value: True to enable symbolic mode, False to disable it.
    """
    _symbolic_mode.set(value)


def in_symbolic_mode():
    """
    Check whether symbolic construction mode is currently active.

    :returns: True if symbolic mode is enabled, otherwise False.
    """
    return _symbolic_mode.get()


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
            if v._node_.parent is not None and isinstance(v, (HasDomain, Source)):
                children[k] = self._copy_child_expression_(v)
            children[k]._node_.parent = self._node_
        return tuple(children.values())

    def _copy_child_expression_(self, child: Optional[SymbolicExpression] = None) -> SymbolicExpression:
        if child is None:
            child = self._child_
        child_cp = child.__new__(child.__class__)
        child_cp.__dict__.update(child.__dict__)
        child_cp._create_node_(child._node_.name + f"_{self._id_}")
        return child_cp

    def _create_node_(self, name: str):
        self._node_ = Node(name)
        self._node_._expression_ = self

    @abstractmethod
    def _reset_cache_(self) -> None:
        """
        Reset the cache of the symbolic expression.
        This method should be implemented by subclasses.
        """
        ...

    @property
    def _yield_when_false_(self) -> bool:
        return self._yield_when_false__

    @_yield_when_false_.setter
    def _yield_when_false_(self, value):
        self._yield_when_false__ = value
        for child in self._children_:
            child._yield_when_false_ = value

    def _get_var_id_(self, operand: SymbolicExpression) -> int:
        if isinstance(operand, An):
            return operand._parent_variable_._id_
        else:
            return operand._id_

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

    def _reset_cache_(self) -> None:
        ...

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
class ResultQuantifier(SymbolicExpression[T], ABC):
    """
    Base for quantifiers that return concrete results from entity/set queries
    (e.g., An, The).
    """
    _child_: QueryObjectDescriptor[T]

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
        self._child_._reset_cache_()

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        return HashedIterable()

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_

    def _process_result_(self, result: Dict[int, HashedValue]) -> Union[T, Dict[SymbolicExpression[T], T]]:
        if isinstance(self._child_, Entity):
            return result[self._child_.selected_variable._id_].value
        elif isinstance(self._child_, SetOf):
            return {self._id_expression_map_[var_id]: v.value for var_id, v in result.items()}
        else:
            raise NotImplementedError(f"Unknown child type {type(self._child_)}")


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

    def evaluate(self) -> Union[Iterable[T], T, Dict[SymbolicExpression[T], T]]:
        result = self._evaluate__()
        result = self._process_result_(result)
        self._reset_cache_()
        return result

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Dict[int, HashedValue]:
        sol_gen = self._child_._evaluate__(sources)
        result = None
        for sol in sol_gen:
            if result is None:
                result = sol
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

    def evaluate(self) -> Union[Iterable[T], Iterable[Dict[SymbolicExpression[T], T]]]:
        results = self._evaluate__()
        yield from map(self._process_result_, results)
        self._reset_cache_()

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[T]:
        values = self._child_._evaluate__(sources)
        for value in values:
            self._is_false_ = self._child_._is_false_
            if self._yield_when_false_ or not self._is_false_:
                yield value


@dataclass(eq=False)
class QueryObjectDescriptor(SymbolicExpression[T], ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """
    _child_: Optional[SymbolicExpression[T]] = field(default=None)
    _warned_vars_: typing.Set = field(default_factory=set, init=False)

    def _evaluate_(self, selected_vars: Optional[Iterable[HasDomain]] = None,
                   sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        seen_values = SeenSet()
        if self._child_:
            child_values = self._child_._evaluate__(sources)
        else:
            child_values = [{}]
        for v in child_values:
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
                for sol in generate_combinations(var_val_gen):
                    v = {var._id_: sol[var][var._id_] for var in selected_vars}
                    if not seen_values.check(v):
                        seen_values.add(v)
                        yield v
            else:
                if not seen_values.check(v):
                    seen_values.add(v)
                    yield v

    def _reset_cache_(self) -> None:
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
                if var not in self._warned_vars_:
                    self._warned_vars_.add(var)
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

    def __repr__(self):
        return self._name_


@dataclass(eq=False)
class SetOf(QueryObjectDescriptor[T]):
    """
    A query over a set of variables.
    """
    selected_variables: Iterable[HasDomain] = field(default_factory=tuple)

    @property
    def _name_(self) -> str:
        return f"SetOf({', '.join(var._name_ for var in self.selected_variables)})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = HashedIterable()
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
            if self.selected_variables:
                yield {var._id_: next(var._evaluate__(sol))[var._id_]
                       for var in self.selected_variables if var._id_ in sol}
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
    selected_variable: Optional[Variable[T]] = field(default=None)
    domain: Optional[Any] = field(default=None)

    def __post_init__(self):
        self.update_variable_domain()
        super().__post_init__()
        self.update_child_expression_with_variable_properties()

    def update_variable_domain(self):
        """
        Update the domain of the variable with the provided entity domain.
        """
        if self.domain is not None:
            if self.selected_variable is not None:
                type_ = self.selected_variable._type_
                child = self.domain
                if not isinstance(child, (Source, HasDomain)):
                    child = Source(type_.__name__, self.domain)
                var_domain = HasType(_child_=child, _type_=(type_,))
                self.selected_variable._update_domain_(var_domain)

    def update_child_expression_with_variable_properties(self):
        """
        Update the child expression with the properties of the variable.
        """
        if self.selected_variable and self.selected_variable._properties_:
            if self._child_:
                self._update_child_(chained_logic(AND, self.selected_variable._properties_, self._child_))
            else:
                self._update_child_(self.selected_variable._properties_)

    @property
    def _name_(self) -> str:
        return f"Entity({self.selected_variable._name_ if self.selected_variable else ''})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = HashedIterable()
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
            if self._yield_when_false_ or not self._is_false_:
                if self.selected_variable:
                    yield from self.selected_variable._evaluate__(sol)
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
class HasDomain(SymbolicExpression[T], ABC):
    _domain_: HashedIterable[Any] = field(default=None, init=False)
    _domain_sources_: Optional[List[Union[HasDomain, Source]]] = field(default_factory=list, init=False)
    _child_: Optional[HasDomain] = field(init=False)

    def __post_init__(self):
        self._update_domain_(self._domain_)
        super().__post_init__()

    def _update_domain_(self, domain):
        if domain is not None:
            if isinstance(domain, (HasDomain, Source)):
                self._domain_sources_.append(domain)
            if isinstance(domain, Source):
                self._domain_ = HashedIterable(domain._evaluate__().value)
            elif isinstance(domain, HasDomain):
                self._domain_ = HashedIterable((next(iter(v.values())) for v in domain._evaluate__()))
            elif is_iterable(domain):
                self._domain_ = HashedIterable(domain)
            else:
                self._domain_ = HashedIterable([HashedValue(domain)])

    def _reset_cache_(self) -> None:
        ...

    def __iter__(self):
        for v in self._domain_:
            yield {self._id_: HashedValue(v)}

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"{self._name_} has no attribute {name}")
        return Attribute(self, name)

    def __call__(self, *args, **kwargs):
        return Call(self, args, kwargs)

    def __eq__(self, other):
        return Comparator(self, other, operator.eq)

    def __contains__(self, item):
        return Comparator(item, self, operator.contains)

    def __ne__(self, other):
        return Comparator(self, other, operator.ne)

    def __lt__(self, other):
        return Comparator(self, other, operator.lt)

    def __le__(self, other):
        return Comparator(self, other, operator.le)

    def __gt__(self, other):
        return Comparator(self, other, operator.gt)

    def __ge__(self, other):
        return Comparator(self, other, operator.ge)

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class DomainFilter(HasDomain, ABC):
    _child_: Union[HasDomain, Source]
    _invert_: bool = field(init=False, default=False)

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) \
            -> Iterable[Dict[int, HashedValue]]:
        child_val = self._child_._evaluate__(sources)
        if self._parent_variable_:
            yield from map(lambda v: {self._parent_variable_._id_: v, self._id_: v},
                           filter(self._filter_func_, child_val))
        else:
            yield from map(lambda v: {self._id_: v},
                           filter(self._filter_func_, child_val))

    def __iter__(self):
        yield from map(lambda v: HashedIterable(values={self._id_: v}),
            filter(self._filter_func_, self._child_._evaluate__()))

    def _filter_func_(self, v: Any) -> bool:
        """
        The filter function to be used to filter the domain, and handle inversion.
        """
        if self._invert_:
            return not self._filter_func__(v)
        return self._filter_func__(v)

    @abstractmethod
    def _filter_func__(self, v: Any) -> bool:
        """
        The filter function to be used to filter the domain.
        """
        ...


@dataclass(eq=False)
class HasType(DomainFilter):
    _type_: Tuple[Type, ...]

    @property
    def _name_(self):
        return f"HasType({','.join(t.__name__ for t in self._type_)})"

    def _filter_func__(self, v: Any) -> bool:
        if isinstance(v, HashedValue):
            v = v.value
        return isinstance(v, self._type_)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_


@dataclass(eq=False)
class Variable(HasDomain[T]):
    _name__: str
    _type_: Type
    _cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    _domain_: Union[HashedIterable, HasDomain, Source, Iterable] = field(default_factory=HashedIterable, kw_only=True)
    _properties_: Optional[SymbolicExpression] = field(default=None)

    def __post_init__(self):
        if self._cls_kwargs_ and not self._type_:
            raise ValueError(f"Variable {self._name_} has class keyword arguments but no type is specified.")
        if type(self) is Variable:
            self._child_ = None
        super().__post_init__()
        if self._cls_kwargs_:
            domain_sources = []
            for k, v in self._cls_kwargs_.items():
                if isinstance(v, HasDomain):
                    domain_sources.append(v)
                else:
                    domain_sources.append(Variable._from_domain_(v, name=self._name_ + '.' + k))
            self._domain_sources_.extend(domain_sources)
            # _ = self._update_children_(*domain_sources)
            self._properties_ = chained_logic(AND, *[getattr(self, k) == v for k, v in self._cls_kwargs_.items()])

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        A variable does not need to evaluate anything by default.
        """
        sources = sources or {}
        if self._id_ in sources:
            yield {self._id_: sources[self._id_]}
        elif self._type_ is not None and not self._domain_:
            kwargs_generators = {k: v._evaluate__(sources) if isinstance(v, SymbolicExpression) else HashedValue([v])
                                 for k, v in self._cls_kwargs_.items()}
            for kwargs in generate_combinations(kwargs_generators):
                instance = self._type_(**{k: v[self._get_var_id_(self._cls_kwargs_[k])].value for k, v in kwargs.items()})
                values = {self._id_: HashedValue(instance)}
                for k, v in kwargs.items():
                    values.update(v)
                yield {self._id_: HashedValue(instance)}
        else:
            yield from self

    @property
    def _name_(self):
        return self._name__

    @classmethod
    def _from_domain_(cls, iterable, clazz: Optional[Type] = None,
                      name: Optional[str] = None) -> Variable:
        if not is_iterable(iterable):
            iterable = HashedIterable([iterable])
        if not clazz:
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
        if self._cls_kwargs_:
            for v in self._cls_kwargs_.values():
                if isinstance(v, HasDomain):
                    variables.extend(v._all_variable_instances_)
        return variables

    def __repr__(self):
        return self._name_


@dataclass(eq=False)
class DomainMapping(HasDomain, ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """
    _child_: HasDomain
    _invert_: bool = field(init=False, default=False)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_ if not isinstance(self._child_, Variable) \
            else [self._child_]

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) \
            -> Iterable[Dict[int, HashedValue]]:
        yield from self._update_domain_(sources)

    def __iter__(self):
        yield from self._update_domain_()

    def _update_domain_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
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
    seen_parent_values: Dict[bool, SeenSet] = field(default_factory=lambda: {True: SeenSet(), False: SeenSet()},
                                                    init=False)

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
            if is_false and self.is_duplicate_output(output):
                continue
            yield output
        if not entered:
            cache_match_count.values[self._node_.name] += 1
        cache_enter_count.values[self._node_.name] = cache.enter_count
        cache_search_count.values[self._node_.name] = cache.search_count

    def is_duplicate_output(self, output: Dict[int, HashedValue]) -> bool:
        required_vars = self._parent_._required_variables_from_child_(self, when_true=not self._is_false_)
        if not required_vars:
            return False
        required_output = {k: v for k, v in output.items() if k in required_vars}
        if self.seen_parent_values[not self._is_false_].check(required_output):
            return True
        else:
            self.seen_parent_values[not self._is_false_].add(required_output)
            return False

    def update_cache(self, values: Dict[int, HashedValue], cache: Optional[IndexedCache] = None):
        if not is_caching_enabled():
            return
        cache = self._cache_ if cache is None else cache
        cache.insert({k: v for k, v in values.items() if k in cache.keys}, output=self._is_false_)

    def _reset_cache_(self) -> None:
        self._cache_.clear()
        self.seen_parent_values[True].clear()
        self.seen_parent_values[False].clear()
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


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """
    left: HasDomain
    right: HasDomain
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
            operand_value_map = {first_operand._id_: first_value[self._get_var_id_(first_operand)]}
            second_values = second_operand._evaluate__(sources)
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

    def get_first_second_operands(self, sources: Dict[int, HashedValue]) -> Tuple[SymbolicExpression, SymbolicExpression]:
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
                if self.is_duplicate_output(left_value):
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
                if self._is_false_ and self.is_duplicate_output(output):
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
                    if self._is_false_ and self.is_duplicate_output(output):
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
