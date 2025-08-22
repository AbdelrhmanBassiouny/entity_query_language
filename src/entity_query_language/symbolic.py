from __future__ import annotations

import contextvars
import operator
import typing
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, ClassVar, Union, Generic, TypeVar
from typing_extensions import dataclass_transform, List, Tuple, Callable

from .enums import RDREdge
from .failures import MultipleSolutionFound
from .utils import make_list, IDGenerator, SeenSet, is_iterable, render_tree

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


@dataclass
class HashedValue(Generic[T]):
    value: T
    id_: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.id_ is None:
            if hasattr(self.value, "_id_"):
                self.id_ = self.value._id_
            else:
                self.id_ = id(self.value)

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other):
        return self.id_ == other.id_


@dataclass
class HashedIterable(Generic[T]):
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """
    iterable: Iterable[HashedValue[T]] = field(default_factory=list)
    values: Dict[int, HashedValue[T]] = field(default_factory=dict)

    def __post_init__(self):
        if self.iterable and not isinstance(self.iterable, HashedIterable):
            self.iterable = (HashedValue(id_=k, value=v) if not isinstance(v, HashedValue) else v
                             for k, v in enumerate(self.iterable))

    def get(self, key: int, default: Any) -> HashedValue[T]:
        return self.values.get(key, default)

    def add(self, value: Any):
        if not isinstance(value, HashedValue):
            value = HashedValue(value)
        if value.id_ not in self.values:
            self.values[value.id_] = value
        return self

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        yield from self.values.values()
        for v in self.iterable:
            self.values[v.id_] = v
            yield v

    def __or__(self, other) -> HashedIterable[T]:
        return self.union(other)

    def __and__(self, other) -> HashedIterable[T]:
        return self.intersection(other)

    def intersection(self, other):
        common_keys = self.values.keys() & other.values.keys()
        common_values = {k: self.values[k] for k in common_keys}
        return HashedIterable(values=common_values)

    def difference(self, other):
        common_keys = self.values.keys() & other.values.keys()
        left_keys = self.values.keys() - other.values.keys()
        values = {k: self.values[k] for k in left_keys}
        return HashedIterable(values=values)

    def union(self, other):
        all_keys = self.values.keys() | other.values.keys()
        all_values = {k: self.values.get(k, other.values.get(k)) for k in all_keys}
        return HashedIterable(values=all_values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, id_: int) -> HashedValue:
        """
        Get the HashedValue by its id.

        :param id_: The id of the HashedValue to get.
        :return: The HashedValue with the given id.
        """
        return self.values[id_]

    def __setitem__(self, id_: int, value: HashedValue[T]):
        """
        Set the HashedValue by its id.

        :param id_: The id of the HashedValue to set.
        :param value: The HashedValue to set.
        """
        self.values[id_] = value

    def __copy__(self):
        """
        Create a shallow copy of the HashedIterable.

        :return: A new HashedIterable instance with the same values.
        """
        return HashedIterable(values=self.values.copy())

    def __contains__(self, item):
        return item in self.values

    def __hash__(self):
        return hash(tuple(sorted(self.values.keys())))

    def __eq__(self, other):
        keys_are_equal = self.values.keys() == other.values.keys()
        if not keys_are_equal:
            return False
        values_are_equal = all(my_v == other_v for my_v, other_v in zip(self.values.values(), other.values.values()))
        return values_are_equal

    def __bool__(self):
        return bool(self.values) or bool(self.iterable)


id_generator = IDGenerator()


class CacheSearchCount:
    val: int = 0

    def update(self):
        self.val += 1


_cache_search_count = CacheSearchCount()


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
        if self._child_._node_.parent is not None:
            child_cp = self._copy_child_expression_()
            self._child_ = child_cp
        self._child_._node_.parent = self._node_

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
        self._node_.weight = RDREdge.Then
        current_parent = SymbolicExpression._current_parent_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._parent_ = current_parent
        self._parent_._conclusion_.add(self)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return []

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

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.value, SymbolicExpression):
            self.value = Variable._from_domain_(self.value)

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
            for k, v in self._cls_kwargs_.items():
                if isinstance(v, HasDomain):
                    self._domain_sources_.append(v)
                else:
                    self._domain_sources_.append(Variable._from_domain_(v, name=self._name_ + '.' + k))

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
        return [self]

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
    left: HasDomain
    right: HasDomain
    _child_: SymbolicExpression = field(init=False, default=None)
    _parent_required_variables__: HashedIterable[Variable] = field(init=False, default_factory=HashedIterable)

    def __post_init__(self):
        if not isinstance(self.left, SymbolicExpression):
            self.left = Variable._from_domain_([self.left])
        if not isinstance(self.right, SymbolicExpression):
            self.right = Variable._from_domain_([self.right])
        super().__post_init__()
        for i, operand in enumerate([self.left, self.right]):
            if operand._node_.parent is not None and isinstance(operand, HasDomain):
                operand = self._copy_child_expression_(operand)
                if i == 0:
                    self.left = operand
                else:
                    self.right = operand
            operand._node_.parent = self._node_
        if isinstance(self.left, BinaryOperator):
            self.left._parent_required_variables__ = self.right._unique_variables_

    @property
    @lru_cache(maxsize=None)
    def _parent_required_variables_(self):
        if self._parent_ is None or isinstance(self._parent_, QueryObjectDescriptor):
            return HashedIterable()
        else:
            return self._parent_required_variables__.union(self._parent_._parent_required_variables_)

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

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[HashedIterable]:
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
        if self.right._parent_variable_._id_ not in sources:
            left_values = self.left._evaluate__(sources)
            for left_value in left_values:
                right_values = self.right._evaluate__(sources)
                for right_value in right_values:
                    res = self.check(left_value, right_value)
                    if res:
                        self._is_false_ = False
                        yield res
                    elif self._yield_when_false_:
                        self._is_false_ = True
                        yield self.get_result_domain(left_value, right_value)

        else:
            right_values = self.right._evaluate__(sources)
            for right_value in right_values:
                left_values = self.left._evaluate__(sources)
                for left_value in left_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

    def check(self, left_value: HashedValue, right_value: HashedValue) -> Optional[HashedIterable]:
        satisfied = self.operation(left_value.value, right_value.value)
        if satisfied:
            return self.get_result_domain(left_value, right_value)
        else:
            return None

    def get_result_domain(self, left_value: HashedValue, right_value: HashedValue) -> HashedIterable:
        left_leaf_value = self.left._parent_variable_._domain_[left_value.id_]
        right_leaf_value = self.right._parent_variable_._domain_[right_value.id_]
        return HashedIterable(values={self.left._parent_variable_._id_: left_leaf_value,
                                      self.right._parent_variable_._id_: right_leaf_value})


@dataclass(eq=False)
class LogicalOperator(BinaryOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions.
    """

    seen_parent_values: SeenSet = field(default_factory=SeenSet, init=False)
    output_cache: Dict[Tuple, bool] = field(default_factory=dict, init=False)

    @property
    def _name_(self):
        return self.__class__.__name__

    def update_output_cache(self, right_value, values_for_right_leaves):
        right_only_values = {k: v for k, v in right_value.values.items() if k in self.right._unique_variables_}
        values_for_right_leaves.update(right_only_values)
        self.output_cache[tuple(sorted(values_for_right_leaves.items()))] = self._is_false_

    def yield_values_from_cache_and_update_seen_parent_values(self, values_for_right_leaves,
                                                              left_value, parent_leaf_ids):
        global _cache_search_count
        for cached_k, output in self.output_cache.items():
            cached_k_dict = dict(cached_k)
            common_ids = values_for_right_leaves.keys() & cached_k_dict.keys()
            _cache_search_count.update()
            if any(values_for_right_leaves[id_] != cached_k_dict[id_] for id_ in common_ids):
                continue
            self._is_false_ = output
            cached_output_values = HashedIterable(values=cached_k_dict)
            output = left_value.union(cached_output_values)
            output_for_parent = {k: v for k, v in output.values.items() if k in parent_leaf_ids}

            if not self.seen_parent_values.check(output_for_parent):
                self.seen_parent_values.add(output_for_parent)
                yield output

    def update_seen_parent_values(self, output: HashedIterable, parent_leaf_ids: Set[int]) -> None:
        output_for_parent = {k: v for k, v in output.values.items() if k in parent_leaf_ids}
        if not self.seen_parent_values.check(output_for_parent):
            self.seen_parent_values.add(output_for_parent)


@dataclass(eq=False)
class AND(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    seen_left_values: SeenSet = field(default_factory=SeenSet, init=False)
    seen_parent_values: SeenSet = field(default_factory=SeenSet, init=False)
    output_cache: Dict[Tuple, bool] = field(default_factory=dict, init=False)

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        # init an empty source if none is provided
        sources = sources or HashedIterable()
        right_values_leaf_ids = [leaf.id_ for leaf in self.right._unique_variables_]
        parent_leaf_ids = [leaf.id_ for leaf in self._parent_required_variables_]


        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value = left_value.union(sources)
            if self._yield_when_false_ and self.left._is_false_:
                self._is_false_ = True
                yield left_value
                self.update_seen_parent_values(left_value, parent_leaf_ids)
                continue
            values_for_right_leaves = {k: left_value[k] for k in right_values_leaf_ids if k in left_value}
            if self.seen_left_values.check(values_for_right_leaves):
                yield from self.yield_values_from_cache_and_update_seen_parent_values(values_for_right_leaves,
                                                                                      left_value, parent_leaf_ids)
                continue
            else:
                self.seen_left_values.add(values_for_right_leaves)

            # constrain right values by available sources
            right_values = self.right._evaluate__(left_value)

            # For the found left value, find all right values,
            # and yield the (left, right) results found.
            for right_value in right_values:
                if self._yield_when_false_:
                    if self.right._is_false_:
                        self._is_false_ = True
                    else:
                        self._is_false_ = False
                output = left_value.union(right_value)
                yield output
                self.update_seen_parent_values(output, parent_leaf_ids)
                self.update_output_cache(right_value, values_for_right_leaves)


@dataclass(eq=False)
class OR(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # init an empty source if none is provided
        sources = sources or HashedIterable()
        seen_values = set()

        # constrain left values by available sources
        for operand in [self.left, self.right]:

            operand_values = operand._evaluate__(sources)

            for operand_value in operand_values:

                # Check operand value, if result is False, continue to next operand value.
                if operand_value not in seen_values:
                    seen_values.add(operand_value)
                    yield sources.union(operand_value)


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
class ExceptIf(LogicalOperator):

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
class XOR(LogicalOperator):
    """
    A symbolic single choice operation that can be used to choose between multiple symbolic expressions.
    """
    seen_parent_values: SeenSet = field(default_factory=SeenSet, init=False)
    output_cache: Dict[Tuple, bool] = field(default_factory=dict, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.left._yield_when_false_ = True
        for child in self.left._descendants_:
            child._yield_when_false_ = True

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle ElseIf logic.
        """
        # init an empty source if none is provided
        sources = sources or HashedIterable()
        seen_values = set()
        seen_right_values = set()
        seen_negative_values = SeenSet()
        shared_ids = list(map(lambda v: v.value._id_,
                         self.left._unique_variables_.intersection(self.right._unique_variables_)))
        parent_leaf_ids = [var.id_ for var in self._parent_required_variables_]

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources)
        for left_value in left_values:
            left_value = left_value.union(sources)
            if self.left._is_false_:
                values_for_right_leaves = {k: left_value[k] for k in shared_ids if k in left_value}
                if seen_negative_values.check(values_for_right_leaves):
                    yield from self.yield_values_from_cache_and_update_seen_parent_values(values_for_right_leaves,
                                                                                          left_value, parent_leaf_ids)
                    continue
                seen_negative_values.add(values_for_right_leaves)
                right_values = self.right._evaluate__(left_value)
                for right_value in right_values:
                    output = None
                    if not self.right._is_false_:
                        self._is_false_ = False
                        output = self.update_conclusion_and_return_operand_value(self.right, right_value,
                                                                                 left_value,
                                                                                 seen_right_values)
                    elif self._yield_when_false_:
                        self._is_false_ = True
                        output = left_value.union(right_value)
                    if output is not None:
                        yield output
                        self._conclusion_.clear()
                    self.update_seen_parent_values(output, parent_leaf_ids)
                    self.update_output_cache(right_value, values_for_right_leaves)
            else:
                self._is_false_ = False
                output = self.update_conclusion_and_return_operand_value(self.left, left_value, sources, seen_values)
                if output is not None:
                    yield output
                    self._conclusion_.clear()
                self.update_seen_parent_values(output, parent_leaf_ids)

    def update_conclusion_and_return_operand_value(self, operand: SymbolicExpression, operand_value: HashedIterable,
                                                   sources: Optional[HashedIterable],
                                                   seen_values: typing.Set)\
            -> Optional[HashedIterable]:
        """
        Evaluate the operand of the XOR operation and yield the results.
        """
        if operand_value not in seen_values:
            self._conclusion_.update(operand._conclusion_)
            seen_values.add(operand_value)
            return sources.union(operand_value)


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
