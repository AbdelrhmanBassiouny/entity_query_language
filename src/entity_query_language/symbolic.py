from __future__ import annotations

import contextvars
import operator
import typing
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import FrozenSet

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, ClassVar, Union, Generic, TypeVar
from typing_extensions import dataclass_transform, List, Tuple, Callable

from .cache_data import _cache_enter_count, _cache_search_count, _cache_match_count, _cache_lookup_time, \
    _cache_update_time, is_caching_enabled
from .enums import RDREdge
from .failures import MultipleSolutionFound
from .utils import make_list, IDGenerator, SeenSet, IndexedCache, is_iterable, render_tree, HashedValue, \
    HashedIterable
import time

_symbolic_mode = contextvars.ContextVar("symbolic_mode", default=False)


def _set_symbolic_mode(value: bool):
    _symbolic_mode.set(value)


def in_symbolic_mode():
    return _symbolic_mode.get()


@dataclass
class SymbolicRule:
    query: Optional[SymbolicExpression] = None

    def __enter__(self):
        if self.query is not None:
            self.query.__enter__()
        _set_symbolic_mode(True)
        return self  # optional, depending on whether you want to assign `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.query is not None:
            self.query.__exit__(exc_type, exc_val, exc_tb)
        _set_symbolic_mode(False)


@dataclass_transform()
def symbol(cls):
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            if len(args) > 0:
                return Variable(args[0], symbolic_cls, _domain_=args[1])
            return Variable(symbolic_cls.__name__, symbolic_cls, _cls_kwargs_=kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls


T = TypeVar("T")

id_generator = IDGenerator()


@dataclass(eq=False)
class SymbolicExpression(Generic[T], ABC):
    _child_: Optional[SymbolicExpression] = field(init=False)
    _id_: int = field(init=False, repr=False, default=None)
    _node_: Node = field(init=False, default=None, repr=False)
    _id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}
    _conclusion_: typing.Set[Conclusion] = field(init=False, default_factory=set)
    _symbolic_expression_stack_: ClassVar[List[SymbolicExpression]] = []
    _yield_when_false_: bool = field(init=False, repr=False, default=False)
    _is_false_: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self._id_ = id_generator(self)
        node_name = self._name_ + f"_{self._id_}"
        self._create_node_(node_name)
        if hasattr(self, "_child_") and self._child_ is not None:
            self._update_child_()
        if self._id_ not in self._id_expression_map_:
            self._id_expression_map_[self._id_] = self

    def _update_child_(self):
        self._child_ = self._update_children_(self._child_)[0]

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
    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[Union[HashedIterable, HashedValue]]:
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

    def _render_tree_(self):
        render_tree(self._root_._node_, True, view=True)

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
        return self._all_variable_instances_[0]

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

    def __xor__(self, other) -> SymbolicExpression:
        return XOR(self, other)

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return OR(self, other)

    def __invert__(self):
        return Not(self)

    def __enter__(self):
        node = self
        if isinstance(node, (ResultQuantifier, Entity)):
            node = node._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(node)
        return self

    def __exit__(self, *args):
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class Source(SymbolicExpression[T]):
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

    @property
    def _source_(self) -> Source:
        return self._node_.leaves[0]._expression_

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> T:
        return self._value_

    def __getattr__(self, name):
        return SourceAttribute(name, getattr(self._value_, name), _child_=self)

    def __call__(self, *args, **kwargs):
        return SourceCall(self._name_, self._value_(*args, **kwargs), self, args, kwargs)


@dataclass(eq=False)
class SourceAttribute(Source):
    _child_: Source = field(kw_only=True)

    @property
    def _name_(self) -> str:
        return f"{self._child_._name_}.{self._name__}"


@dataclass(eq=False)
class SourceCall(Source):
    _child_: Source
    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs_: Dict[str, Any] = field(default_factory=dict)

    @property
    def _name_(self) -> str:
        return (f"{self._child_._name_}({', '.join(map(str, self._args_))},"
                f" {', '.join(f'{k}={v}' for k, v in self._kwargs_.items())})")


@dataclass(eq=False)
class ResultQuantifier(SymbolicExpression[T], ABC):
    _child_: Entity[T]

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}()"

    def evaluate(self) -> Iterable[T]:
        return self._evaluate__()

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        return HashedIterable()

    def except_if(self, *conditions: SymbolicExpression[T]) -> ResultQuantifier[T]:
        """
        Exclude results that match the given conditions.
        """
        new_branch = chained_logic(AND, *conditions)
        new_conditions_root = self._conditions_root_ & new_branch
        new_branch._node_.weight = RDREdge.Refinement
        new_conditions_root._node_.parent = self._child_._node_
        return self

    def else_if(self, *conditions: SymbolicExpression[T]) -> ResultQuantifier[T]:
        new_branch = chained_logic(AND, *conditions)
        new_conditions_root = self._conditions_root_ ^ new_branch
        new_branch._node_.weight = RDREdge.Alternative
        new_conditions_root._node_.parent = self._child_._node_
        return self

    def also_if(self, *conditions: SymbolicExpression[T]) -> ResultQuantifier[T]:
        new_branch = chained_logic(OR, *conditions)
        new_conditions_root = self._conditions_root_ | new_branch
        new_branch._node_.weight = RDREdge.Next
        new_conditions_root._node_.parent = self._child_._node_
        return self

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_


@dataclass(eq=False)
class Conclusion(SymbolicExpression[T], ABC):
    var: HasDomain
    value: Any
    _child_: Optional[SymbolicExpression[T]] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()

        self.var, self.value = self._update_children_(self.var, self.value)

        self._node_.weight = RDREdge.Then

        current_parent = SymbolicExpression._current_parent_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._parent_ = current_parent
        self._parent_._add_conclusion_(self)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self.var._all_variable_instances_ + self.value._all_variable_instances_

    @property
    def _name_(self) -> str:
        value_str = self.value._type_.__name__ if isinstance(self.value, Variable) else str(self.value)
        return f"{self.__class__.__name__}({self.var._name_}, {value_str})"


@dataclass(eq=False)
class Set(Conclusion[T]):

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> HashedIterable:
        if self._parent_._id_ not in sources:
            parent_value = next(iter(self._parent_._evaluate__(sources)))
        else:
            parent_value = sources[self._parent_._id_]
        parent_value.value = self.value
        return sources


@dataclass(eq=False)
class Add(Conclusion[T]):

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> HashedIterable:
        v = next(iter(self.value._evaluate__(sources)))
        self.var._domain_[v.id_] = v
        sources[self.var._parent_variable_._id_] = v
        return sources


@dataclass(eq=False)
class The(ResultQuantifier[T]):

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> T:
        sol_gen = self._child_._evaluate__(sources)
        first_val = next(sol_gen)
        try:
            second_val = next(sol_gen)
        except StopIteration:
            return first_val
        else:
            raise MultipleSolutionFound(first_val, second_val)


@dataclass(eq=False)
class An(ResultQuantifier[T]):

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[T]:
        yield from self._child_._evaluate__(sources)


@dataclass(eq=False)
class QueryObjectDescriptor(SymbolicExpression[T], ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """
    _child_: SymbolicExpression[T]

    def _evaluate_(self, selected_vars: Iterable[HasDomain],
                   sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        if isinstance(selected_vars, HasDomain):
            selected_vars = [selected_vars]
        seen_values = set()
        for v in self._child_._evaluate__(sources):
            for conclusion in self._child_._conclusion_:
                v = conclusion._evaluate__(v)
            for var in selected_vars:
                if var._id_ not in v:
                    v[var._id_] = next(var._evaluate__(v))
            v = HashedIterable(values={var._id_: v[var._id_] for var in selected_vars})
            if v not in seen_values:
                seen_values.add(v)
                yield v

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self._child_._all_variable_instances_


@dataclass(eq=False)
class SetOf(QueryObjectDescriptor[T]):
    """
    A query over a set of variables.
    """
    selected_variables_: Iterable[HasDomain]

    @property
    def _name_(self) -> str:
        return f"SetOf({', '.join(var._name_ for var in self.selected_variables_)})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = HashedIterable()
        required_vars.update(self.selected_variables_)
        for var in self.selected_variables_:
            required_vars.update(var._unique_variables_)
        for conc in child._conclusion_:
            required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[Dict[HasDomain, Any]]:
        sol_gen = self._evaluate_(self.selected_variables_, sources)
        for sol in sol_gen:
            yield {var: sol[var._id_].value for var in self.selected_variables_ if var._id_ in sol}


@dataclass(eq=False)
class Entity(QueryObjectDescriptor[T]):
    """
    A query over a single variable.
    """
    selected_variable_: T

    @property
    def _name_(self) -> str:
        return f"Entity({self.selected_variable_._name_})"

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = HashedIterable()
        required_vars.add(self.selected_variable_)
        required_vars.update(self.selected_variable_._unique_variables_)
        for conc in child._conclusion_:
            required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[T]:
        sol_gen = self._evaluate_(self.selected_variable_)
        for sol in sol_gen:
            yield sol[self.selected_variable_._id_].value


@dataclass(eq=False)
class HasDomain(SymbolicExpression, ABC):
    _domain_: HashedIterable[Any] = field(default=None, init=False)
    _domain_sources_: Optional[List[Union[HasDomain, Source]]] = field(default_factory=list, init=False)
    _child_: Optional[HasDomain] = field(init=False)

    def __post_init__(self):
        if self._domain_ is not None:
            if isinstance(self._domain_, (HasDomain, Source)):
                self._domain_sources_.append(self._domain_)
            if isinstance(self._domain_, Source):
                self._domain_ = HashedIterable(self._domain_._evaluate__().value)
            elif isinstance(self._domain_, HasDomain):
                self._domain_ = HashedIterable(self._domain_._evaluate__())
            elif is_iterable(self._domain_):
                self._domain_ = HashedIterable(self._domain_)
            else:
                self._domain_ = HashedIterable([HashedValue(self._domain_)])
        super().__post_init__()

    @property
    def _source_(self) -> HasDomain:
        return self._node_.leaves[0]._expression_

    def __iter__(self):
        yield from self._domain_

    def __getattr__(self, name):
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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) \
            -> Iterable[Union[HashedIterable, HashedValue]]:
        child_val = self._child_._evaluate__(sources)
        if (self._conditions_root_ is self) or isinstance(self._parent_, LogicalOperator):
            yield from map(lambda v: HashedIterable(values={self._parent_variable_._id_: v}),
                           filter(self._filter_func_, child_val))
        else:
            yield from filter(self._filter_func_, child_val)

    def __iter__(self):
        yield from filter(self._filter_func_, self._child_._evaluate__())

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
    _type_: Tuple[Type,...]

    @property
    def _name_(self):
        return f"HasType({self._type_.__name__})"

    def _filter_func__(self, v: Any) -> bool:
        if isinstance(v, HashedValue):
            v = v.value
        return isinstance(v, self._type_)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return []


@dataclass(eq=False)
class Variable(HasDomain):
    _name__: str
    _type_: Type
    _cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    _domain_: Union[HashedIterable, HasDomain, Source, Iterable] = field(default_factory=HashedIterable, kw_only=True)

    def __post_init__(self):
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
            _ = self._update_children_(*domain_sources)

    def _evaluate__(self, sources: Optional[HashedIterable[Any]] = None) -> Iterable[HashedValue]:
        """
        A variable does not need to evaluate anything by default.
        """
        sources = sources or HashedIterable()
        if self._id_ in sources:
            yield from (sources[self._id_],)
        elif not self._domain_:
            def domain_gen():
                cls_kwargs = {k: v._evaluate__(sources) if isinstance(v, HasDomain) else v for k, v in
                              self._cls_kwargs_.items()}
                symbolic_vars = []
                for k, v in self._cls_kwargs_.items():
                    if isinstance(v, HasDomain):
                        symbolic_vars.append(v)
                while True:
                    try:
                        instance = self._type_(**{k: next(v).value if k in symbolic_vars else v.value
                                                  for k, v in cls_kwargs.items()})
                        yield HashedValue(instance)
                    except StopIteration:
                        break
            yield from domain_gen()
        else:
            yield from self

    @property
    def _name_(self):
        return self._name__

    @classmethod
    def _from_domain_(cls, iterable, clazz: Optional[Type] = None,
                      name: Optional[str] = None) -> Variable:
        if not is_iterable(iterable):
            iterable = make_list(iterable)
        if not clazz:
            clazz = type(next((iter(iterable)), None))
        if name is None:
            name = clazz.__name__
        return Variable(name, clazz, _domain_=iterable)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        variables =  [self]
        if self._cls_kwargs_:
            for v in self._cls_kwargs_.values():
                if isinstance(v, HasDomain):
                    variables.extend(v._all_variable_instances_)
        return variables

    def __repr__(self):
        return (f"Symbolic({self._type_.__name__}("
                f"{', '.join(f'{k}={v!r}' for k, v in self._cls_kwargs_.items())}))")


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
        return [next(iter(self._unique_variables_)).value]

    @property
    @lru_cache(maxsize=None)
    def _unique_variables_(self) -> HashedIterable[Variable]:
        child = self._child_
        while not isinstance(child, Variable):
            child = child._child_
        return HashedIterable(values={child._id_: HashedValue(child)})

    def _evaluate__(self, sources: Optional[HashedIterable] = None) \
            -> Iterable[Union[HashedIterable, HashedValue]]:
        child_val = self._child_._evaluate__(sources)
        if (self._conditions_root_ is self) or isinstance(self._parent_, LogicalOperator):
            for child_v in child_val:
                v = self._apply_mapping_(child_v)
                if (not self._invert_ and v.value) or (self._invert_ and not v.value):
                    yield HashedIterable(values={self._parent_variable_._id_: self._parent_variable_._domain_[v.id_]})
        else:
            yield from (self._apply_mapping_(v) for v in child_val)

    def __iter__(self):
        yield from (self._apply_mapping_(v) for v in self._child_)

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
class ConstrainingOperator(SymbolicExpression, ABC):
    """
    An abstract base class for operators that can constrain symbolic expressions.
    This is used to ensure that the operator can be applied to symbolic expressions
    and that it can constrain the results based on indices.
    """
    ...


@dataclass(eq=False)
class BinaryOperator(ConstrainingOperator, ABC):
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
        for values, is_false in cache.retrieve(variables_sources):
            entered = True
            self._is_false_ = is_false
            _cache_match_count.values[self._node_.name] += 1
            yield HashedIterable(values=values)
        if not entered:
            _cache_match_count.values[self._node_.name] += 1
        _cache_enter_count.values[self._node_.name] = cache.enter_count
        _cache_search_count.values[self._node_.name] = cache.search_count

    def update_cache(self, values: HashedIterable, cache: Optional[IndexedCache] = None):
        if not is_caching_enabled():
            return
        cache = self._cache_ if cache is None else cache
        cache.insert({k: v for k, v in values.values.items() if k in cache.keys}, output=self._is_false_)

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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Compares the left and right symbolic variables using the "operation".

        :param sources: Dictionary of symbolic variable id to a value of that variable, the left and right values
        will retrieve values from sources if they exist, otherwise will directly retrieve them from the original
        sources.
        :return: Dictionary of symbolic variable id to a value of that variable, it will contain only two values,
        the left and right symbolic values.
        :raises StopIteration: If one of the left or right values are being retrieved directly from the original
        source and the source has been exhausted.
        """
        sources = sources or HashedIterable()

        if is_caching_enabled():
            if self._cache_.check(sources.values):
                yield from self.yield_from_cache(sources.values)
                return

        order_operand_map = self.get_operand_order_map(sources)

        first_values = order_operand_map['first']._evaluate__(sources)
        for first_value in first_values:
            operand_value_map = {order_operand_map['first']: first_value}
            second_values = order_operand_map['second']._evaluate__(sources)
            for second_value in second_values:
                operand_value_map[order_operand_map['second']] = second_value
                res = self.apply_operation(operand_value_map)
                self._is_false_ = not res
                if res or self._yield_when_false_:
                    values = self.get_result_domain(operand_value_map)
                    self.update_cache(values)
                    yield values

    def apply_operation(self, operand_values: Dict[HasDomain, HashedValue]):
        return self.operation(operand_values[self.left].value, operand_values[self.right].value)

    def get_operand_order_map(self, sources: HashedIterable):
        if self.right._parent_variable_._id_ in sources:
            return {'first': self.right, 'second': self.left}
        else:
            return {'first': self.left, 'second': self.right}

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
    seen_parent_values: Dict[bool, SeenSet] = field(default_factory=lambda: {True: SeenSet(), False: SeenSet()},
                                                    init=False)

    def __post_init__(self):
        super().__post_init__()
        self.right_cache.keys = [v.id_ for v in self.right._unique_variables_]

    @property
    def _name_(self):
        return self.__class__.__name__

    def is_duplicate_output(self, output: HashedIterable) -> bool:
        required_vars = self._parent_._required_variables_from_child_(self, when_true=not self._is_false_)
        if not required_vars:
            return False
        required_output = {k: v for k, v in output.values.items() if k in required_vars}
        if self.seen_parent_values[not self._is_false_].check(required_output):
            return True
        else:
            self.seen_parent_values[not self._is_false_].add(required_output)
            return False

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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        # init an empty source if none is provided
        sources = sources or HashedIterable()

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value = left_value.union(sources)
            if self._yield_when_false_ and self.left._is_false_:
                self._is_false_ = True
                yield left_value
                continue

            if is_caching_enabled() and self.right_cache.check(left_value.values):
                yield from self.yield_from_cache(left_value.values, self.right_cache)
                continue

            # constrain right values by available sources
            right_values = self.right._evaluate__(left_value)

            # For the found left value, find all right values,
            # and yield the (left, right) results found.
            for right_value in right_values:
                output = right_value.union(left_value)
                self._is_false_ = self.right._is_false_
                self.update_cache(right_value, self.right_cache)
                yield output


@dataclass(eq=False)
class OR(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    seen_values: SeenSet = field(default_factory=SeenSet, init=False)

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        required_vars = self._parent_._required_variables_from_child_(self, when_true)
        for conc in self._conclusion_:
            required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # init an empty source if none is provided
        sources = sources or HashedIterable()

        # constrain left values by available sources
        for operand in [self.left, self.right]:

            operand_values = operand._evaluate__(sources)

            for operand_value in operand_values:
                required_vars = self._required_variables_from_child_(operand, when_true=not self._is_false_)
                output_for_parent = {k: v for k, v in operand_value.values.items() if k in required_vars}
                if self.seen_values.check(output_for_parent):
                    continue
                self.seen_values.add(output_for_parent)
                yield sources.union(operand_value)


@dataclass(eq=False)
class ConclusionSelector(LogicalOperator, ABC):
    concluded_before: Dict[bool, SeenSet] = field(default_factory=lambda: {True: SeenSet(), False: SeenSet()},
                                                  init=False)

    def update_conclusion(self, output: HashedIterable, conclusions: typing.Set[Conclusion]) -> None:
        if not conclusions:
            return
        required_vars = HashedIterable()
        for conclusion in conclusions:
            required_vars.update(conclusion._unique_variables_)
        required_output = {k: v for k, v in output.values.items() if k in required_vars}
        if not self.concluded_before[not self._is_false_].check(required_output):
            self._conclusion_.update(conclusions)
            self.concluded_before[not self._is_false_].add(required_output)


def refinement(*conditions: Union[SymbolicExpression[T], bool]) -> SymbolicExpression[T]:
    """
    Exclude results that match the given conditions.
    """
    new_branch = chained_logic(AND, *conditions)
    prev_parent = SymbolicExpression._current_parent_()._parent_
    new_conditions_root = ExceptIf(SymbolicExpression._current_parent_(), new_branch)
    new_branch._node_.weight = RDREdge.Refinement
    new_conditions_root._parent_ = prev_parent
    return new_conditions_root.right


@dataclass(eq=False)
class ExceptIf(ConclusionSelector):

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        if not child:
            child = self.left
        required_vars = HashedIterable()
        when_false = not when_true
        if child is self.left:
            if when_true:
                required_vars.update(self.right._unique_variables_)
            for conc in self.left._conclusion_.union(self.right._conclusion_):
                required_vars.update(conc._unique_variables_)
        elif child is self.right:
            if when_true:
                for conc in self.right._conclusion_:
                    required_vars.update(conc._unique_variables_)
            if when_false and not self.left._is_false_:
                for conc in self.left._conclusion_:
                    required_vars.update(conc._unique_variables_)
        required_vars.update(self._parent_._required_variables_from_child_(self, when_true))
        for conc in self._conclusion_:
            required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """

        # init an empty source if none is provided
        sources = sources or HashedIterable()
        right_values_leaf_ids = [leaf.id_ for leaf in self.right._unique_variables_]
        seen_left_values = SeenSet()

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:

            values_for_right_leaves = {k: left_value[k] for k in right_values_leaf_ids if k in left_value}
            if seen_left_values.check(values_for_right_leaves):
                continue
            else:
                seen_left_values.add(values_for_right_leaves)

            left_value = left_value.union(sources)

            right_yielded = False
            for right_value in self.right._evaluate__(left_value):
                right_yielded = True
                self._conclusion_.update(self.right._conclusion_)
                yield left_value.union(right_value)
                self._conclusion_.clear()
            if not right_yielded:
                self._conclusion_.update(self.left._conclusion_)
                yield left_value
                self._conclusion_.clear()


def alternative(*conditions: Union[SymbolicExpression[T], bool]) -> SymbolicExpression[T]:
    """
    Exclude results that match the given conditions.
    """
    new_branch = chained_logic(AND, *conditions)
    current_node = SymbolicExpression._current_parent_()
    if isinstance(current_node._parent_, XOR):
        current_node = current_node._parent_
    prev_parent = current_node._parent_
    new_conditions_root = XOR(current_node, new_branch)
    new_branch._node_.weight = RDREdge.Alternative
    new_conditions_root._parent_ = prev_parent
    if isinstance(prev_parent, BinaryOperator):
        prev_parent.right = new_conditions_root
    return new_conditions_root.right


@dataclass(eq=False)
class XOR(ConclusionSelector):
    """
    A symbolic single choice operation that can be used to choose between multiple symbolic expressions.
    """

    def __post_init__(self):
        super().__post_init__()
        self.left._yield_when_false_ = True
        for child in self.left._descendants_:
            child._yield_when_false_ = True

    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: Optional[bool] = None):
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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle ElseIf logic.
        """
        # init an empty source if none is provided
        sources = sources or HashedIterable()

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value = left_value.union(sources)
            if self.left._is_false_:
                if is_caching_enabled() and self.right_cache.check(left_value.values):
                    yield from self.yield_from_cache(left_value.values, self.right_cache)
                    continue
                right_values = self.right._evaluate__(left_value)
                for right_value in right_values:
                    self._is_false_ = self.right._is_false_
                    output = left_value.union(right_value)
                    if not self._is_false_:
                        self.update_conclusion(output, self.right._conclusion_)
                    elif not self._yield_when_false_:
                        continue
                    self.update_cache(right_value, self.right_cache)
                    yield output
            else:
                self._is_false_ = False
                self.update_conclusion(left_value, self.left._conclusion_)
                yield left_value
            self._conclusion_.clear()


def Not(operand: Any) -> SymbolicExpression:
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Variable._from_domain_(operand)
    if isinstance(operand, AND):
        operand = OR(Not(operand.left), Not(operand.right))
    elif isinstance(operand, OR):
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
