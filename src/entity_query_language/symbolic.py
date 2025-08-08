from __future__ import annotations

import contextvars
from abc import abstractmethod, ABC
from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, ClassVar, Union, Generic, TypeVar
from typing_extensions import dataclass_transform, List, Tuple

from .utils import is_iterable
from .utils import make_list, IDGenerator

_symbolic_mode = contextvars.ContextVar("symbolic_mode", default=False)


def _set_symbolic_mode(value: bool):
    _symbolic_mode.set(value)


def in_symbolic_mode():
    return _symbolic_mode.get()


class SymbolicMode:
    def __enter__(self):
        _set_symbolic_mode(True)
        return self  # optional, depending on whether you want to assign `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_symbolic_mode(False)


T = TypeVar("T")


@dataclass
class HashedValue(Generic[T]):
    value: T
    id_: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.id_ is None:
            if hasattr(self.value, "id_"):
                self.id_ = self.value.id_
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

    def __copy__(self):
        """
        Create a shallow copy of the HashedIterable.

        :return: A new HashedIterable instance with the same values.
        """
        # iterable_copy, self.iterable = itertools.tee(self.iterable, 2)
        return HashedIterable(values=self.values.copy())  # , iterable=iterable_copy)

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


id_generator = IDGenerator()


@dataclass(eq=False)
class SymbolicExpression(ABC):
    child_: Optional[SymbolicExpression] = field(init=False)
    id_: int = field(init=False, repr=False, default=None)
    node_: Node = field(init=False, default=None, repr=False)
    id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}

    def __post_init__(self):
        self.id_ = id_generator(self)
        node_name = self.name_ + f"_{self.id_}"
        self._create_node_(node_name)
        if self.child_ is not None:
            self._update_child_()
        if self.id_ not in self.id_expression_map_:
            self.id_expression_map_[self.id_] = self

    def _update_child_(self):
        if self.child_.node_.parent is not None:
            child_cp = self._copy_child_expression_()
            self.child_ = child_cp
        self.child_.node_.parent = self.node_

    def _copy_child_expression_(self):
        child_cp = self.child_.__new__(self.child_.__class__)
        child_cp.__dict__.update(self.child_.__dict__)
        child_cp._create_node_(self.child_.node_.name + f"_{self.id_}")
        return child_cp

    def _create_node_(self, name: str):
        self.node_ = Node(name)
        self.node_._expression = self

    def evaluate_(self, selected_vars: Iterable[HasDomain],
                  sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        seen_values = set()
        for v in self.evaluate__(sources):
            v = HashedIterable(values={var.id_: v[var.id_] for var in selected_vars})
            if v not in seen_values:
                seen_values.add(v)
                yield v

    @abstractmethod
    def evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[Union[HashedIterable, HashedValue]]:
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

    @property
    def parent_(self) -> Optional[SymbolicExpression]:
        if self.node_.parent is not None:
            return self.node_.parent._expression
        return None

    @property
    def root_(self) -> SymbolicExpression:
        """
        Get the root of the symbolic expression tree.
        """
        return self.node_.root._expression

    @property
    @abstractmethod
    def name_(self) -> str:
        pass

    @property
    def all_nodes_(self) -> List[SymbolicExpression]:
        return [self] + self.descendants_

    @property
    def all_node_names_(self) -> List[str]:
        return [node.node_.name for node in self.all_nodes_]

    @property
    def descendants_(self) -> List[SymbolicExpression]:
        return [d._expression for d in self.node_.descendants]

    @property
    def children_(self) -> List[SymbolicExpression]:
        return [c._expression for c in self.node_.children]

    def __getattr__(self, name):
        return Attribute(self, name)

    def __call__(self, *args, **kwargs):
        return Call(self, args, kwargs)

    def __eq__(self, other):
        return Comparator(self, '==', other)

    def in_(self, other):
        """
        Check if the symbolic expression is in another iterable or symbolic expression.
        """
        return in_(self, other)

    def contains_(self, item):
        """
        Check if the symbolic expression contains a specific item.
        """
        return self.__contains__(item)

    def __contains__(self, item):
        return Comparator(item, 'in', self)

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return OR(self, other)

    def __invert__(self):
        return Not(self)

    def __ne__(self, other):
        return Comparator(self, '!=', other)

    def __lt__(self, other):
        return Comparator(self, '<', other)

    def __le__(self, other):
        return Comparator(self, '<=', other)

    def __gt__(self, other):
        return Comparator(self, '>', other)

    def __ge__(self, other):
        return Comparator(self, '>=', other)

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class HasDomain(SymbolicExpression, ABC):
    domain_: HashedIterable[Any] = field(default=None, init=False)
    child_: HasDomain = field(init=False)

    def __post_init__(self):
        if self.domain_ is not None:
            self.domain_ = HashedIterable(self.domain_)
        super().__post_init__()

    def __iter__(self):
        yield from self.domain_

    @property
    @lru_cache(maxsize=None)
    def leaf_id_(self):
        return self.leaf_.id_

    @property
    @lru_cache(maxsize=None)
    def leaf_(self) -> HasDomain:
        return list(self.leaves_)[0].value

    @property
    @lru_cache(maxsize=None)
    def leaves_(self) -> HashedIterable[HasDomain]:
        if self.child_ is not None and hasattr(self.child_, 'leaves_'):
            return self.child_.leaves_
        else:
            value = HashedValue(value=self)
            return HashedIterable(values={value.id_: value})

    @property
    @lru_cache(maxsize=None)
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        child_leaves = []
        if self.child_ is not None and hasattr(self.child_, 'all_leaf_instances_'):
            child_leaves = self.child_.all_leaf_instances_
        return [self] + child_leaves


@dataclass(eq=False)
class Variable(HasDomain):
    cls_: Optional[Type] = field(default=None)
    cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    domain_: HashedIterable[Any] = field(default=None, kw_only=True)
    child_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.domain_ is None and self.cls is not None:
            def domain_gen():
                cls_kwarg_gen = {k: iter(v) if is_iterable(v) else v for k, v in self.cls_kwargs_.items()}
                while True:
                    try:
                        instance = self.cls_.__new__(self.cls_)
                        for k, kwarg_gen in cls_kwarg_gen.items():
                            setattr(instance, k, next(kwarg_gen) if is_iterable(kwarg_gen) else kwarg_gen)
                        yield instance
                    except StopIteration:
                        break

            self.domain_: HashedIterable = HashedIterable(domain_gen())

    def evaluate__(self, sources: Optional[HashedIterable[Any]] = None) -> Iterable[HashedValue]:
        """
        A variable does not need to evaluate anything by default.
        """
        sources = sources or HashedIterable()
        if self.id_ in sources:
            yield from (sources[self.id_],)
        else:
            yield from self

    @property
    def name_(self):
        return self.cls_.__name__

    @classmethod
    def from_domain_(cls, iterable, clazz: Optional[Type] = None,
                     child: Optional[SymbolicExpression] = None) -> Variable:
        if not is_iterable(iterable):
            iterable = make_list(iterable)
        if not clazz:
            clazz = type(next((iter(iterable)), None))
        return Variable(clazz, domain_=iterable, child_=child)

    def __repr__(self):
        return (f"Symbolic({self.cls_.__name__}("
                f"{', '.join(f'{k}={v!r}' for k, v in self.cls_kwargs_.items())}))")


@dataclass(eq=False)
class DomainMapping(HasDomain, ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """
    child_: HasDomain
    invert_: bool = field(init=False, default=False)

    def evaluate__(self, sources: Optional[HashedIterable] = None) \
            -> Iterable[Union[HashedIterable, HashedValue]]:
        child_val = self.child_.evaluate__(sources)
        if (self.root_ is self) or isinstance(self.parent_, LogicalOperator):
            for child_v in child_val:
                v = self.apply_(child_v)
                if (not self.invert_ and v.value) or (self.invert_ and not v.value):
                    yield HashedIterable(values={self.leaf_id_: self.leaf_.domain_[v.id_]})
        else:
            yield from (self.apply_(v) for v in child_val)

    def __iter__(self):
        yield from (self.apply_(v) for v in self.child_)

    @abstractmethod
    def apply_(self, value: HashedValue) -> HashedValue:
        """
        Apply the domain mapping to a symbolic value.
        """
        pass


@dataclass(eq=False)
class Attribute(DomainMapping):
    """
    A symbolic attribute that can be used to access attributes of symbolic variables.
    """
    attr_name_: str

    def apply_(self, value: HashedValue) -> HashedValue:
        return HashedValue(id_=value.id_, value=getattr(value.value, self.attr_name_))

    @property
    def name_(self):
        return f"{self.child_.name_}.{self.attr_name_}"


@dataclass(eq=False)
class Call(DomainMapping):
    """
    A symbolic call that can be used to call methods on symbolic variables.
    """
    args_: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs_: Dict[str, Any] = field(default_factory=dict)

    def apply_(self, value: HashedValue) -> HashedValue:
        if len(self.args_) > 0 or len(self.kwargs_) > 0:
            return HashedValue(id_=value.id_, value=value.value(*self.args_, **self.kwargs_))
        else:
            return HashedValue(id_=value.id_, value=value.value())

    @property
    def name_(self):
        return f"{self.child_.name_}()"


@dataclass(eq=False)
class ConstrainingOperator(SymbolicExpression, ABC):
    """
    An abstract base class for operators that can constrain symbolic expressions.
    This is used to ensure that the operator can be applied to symbolic expressions
    and that it can constrain the results based on indices.
    """
    ...


@dataclass(eq=False)
class UnaryOperator(ConstrainingOperator, ABC):
    """
    A base class for unary operators that can be used to apply operations on symbolic expressions.
    """
    operand_: HasDomain

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.operand_, SymbolicExpression):
            self.operand_ = Variable.from_domain_(self.operand_)

    @property
    def name_(self):
        return f"{self.operation} {self.operand_.name}"

    @property
    @lru_cache(maxsize=None)
    def leaves_(self) -> HashedIterable[HasDomain]:
        return self.operand_.leaves_

    @property
    @lru_cache(maxsize=None)
    def all_leaf_instances_(self) -> List[HasDomain]:
        return self.operand_.all_leaf_instances_


@dataclass(eq=False)
class BinaryOperator(ConstrainingOperator, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """
    left_: SymbolicExpression
    operation_: str
    right_: SymbolicExpression
    child_: SymbolicExpression = field(init=False, default=None)
    invert__: bool = field(init=False, default=False)

    def __post_init__(self):
        if not isinstance(self.left_, SymbolicExpression):
            self.left_ = Variable.from_domain_(self.left_)
        if not isinstance(self.right_, SymbolicExpression):
            self.right_ = Variable.from_domain_(self.right_)
        super().__post_init__()
        for operand in [self.left_, self.right_]:
            operand.node_.parent = self.node_

    @property
    def invert_(self):
        return self.invert__

    @invert_.setter
    def invert_(self, value):
        if value == self.invert__:
            return
        self.invert__ = value
        prev_operation = copy(self.operation_)
        match self.operation_:
            case "<":
                self.operation_ = ">=" if self.invert_ else self.operation_
            case ">":
                self.operation_ = "<=" if self.invert_ else self.operation_
            case "<=":
                self.operation_ = ">" if self.invert_ else self.operation_
            case ">=":
                self.operation_ = "<" if self.invert_ else self.operation_
            case "==":
                self.operation_ = "!=" if self.invert_ else self.operation_
            case "!=":
                self.operation_ = "==" if self.invert_ else self.operation_
            case "in":
                self.operation_ = "not in" if self.invert_ else self.operation_
        self.node_.name = self.node_.name.replace(prev_operation, self.operation_)

    @property
    def name_(self):
        return self.operation_

    @property
    @lru_cache(maxsize=None)
    def leaves_(self) -> HashedIterable[HasDomain]:
        return self.left_.leaves_ | self.right_.leaves_

    @property
    @lru_cache(maxsize=None)
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left_.all_leaf_instances_ + self.right_.all_leaf_instances_


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """
    left_: HasDomain
    right_: HasDomain

    def evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[HashedIterable]:
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
        if self.right_.leaf_id_ not in sources:
            left_values = self.left_.evaluate__(sources)
            for left_value in left_values:
                right_values = self.right_.evaluate__(sources)
                for right_value in right_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

        else:
            right_values = self.right_.evaluate__(sources)
            for right_value in right_values:
                left_values = self.left_.evaluate__(sources)
                for left_value in left_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

    def check(self, left_value: HashedValue, right_value: HashedValue) -> Optional[HashedIterable]:
        satisfied = eval(f"left_value.value {self.operation_} right_value.value")
        if satisfied:
            left_leaf_value = self.left_.leaf_.domain_[left_value.id_]
            right_leaf_value = self.right_.leaf_.domain_[right_value.id_]
            return HashedIterable(values={self.left_.leaf_id_: left_leaf_value, self.right_.leaf_id_: right_leaf_value})
        else:
            return None


@dataclass(eq=False)
class LogicalOperator(BinaryOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions.
    """
    operation_: str = field(init=False)

    def __post_init__(self):
        self.operation_ = self.__class__.__name__
        super().__post_init__()


@dataclass(eq=False)
class AND(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:

        # init an empty source if none is provided
        original_sources = sources or HashedIterable()
        sources = copy(original_sources)
        left_sources = copy(sources)
        right_values_leaf_ids = [leaf.id_ for leaf in self.right_.leaves_]
        seen_left_values = set()

        # constrain left values by available sources
        left_values = self.left_.evaluate__(left_sources)
        for left_value in left_values:

            values_for_right_leaves = {(k, left_value[k]) for k in right_values_leaf_ids if k in left_value}
            if seen_left_values and values_for_right_leaves.issubset(seen_left_values):
                continue
            else:
                seen_left_values.update(values_for_right_leaves)

            # update the sources
            sources = sources.union(left_value)

            # constrain right values by available sources
            right_values = self.right_.evaluate__(sources)

            # For the found left value, find all right values,
            # and yield the (left, right) results found.
            for right_value in right_values:

                sources = sources.union(right_value)

                yield sources

            sources = copy(original_sources)


@dataclass(eq=False)
class OR(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # init an empty source if none is provided
        original_sources = sources or HashedIterable()
        seen_values = set()

        # constrain left values by available sources
        for operand in [self.left_, self.right_]:

            operand_sources = copy(original_sources)
            operand_values = operand.evaluate__(operand_sources)

            for operand_value in operand_values:

                # Check operand value, if result is False, continue to next operand value.
                if operand_value not in seen_values:
                    seen_values.add(operand_value)
                    operand_sources = operand_sources.union(operand_value)
                    yield operand_sources

                operand_sources = copy(original_sources)


@dataclass_transform()
def symbol(cls):
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            return Variable(symbolic_cls, cls_kwargs_=kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls


def And(*conditions):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    return chained_logic(AND, *conditions)


def Or(*conditions):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    return chained_logic(OR, *conditions)


def Not(operand: Any) -> SymbolicExpression:
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """
    if not isinstance(operand, SymbolicExpression):
        operand = Variable.from_domain_(operand)
    if isinstance(operand, AND):
        operand = OR(Not(operand.left_), Not(operand.right_))
    elif isinstance(operand, OR):
        operand = AND(Not(operand.left_), Not(operand.right_))
    else:
        operand.invert_ = True
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


def contains(container, item):
    """
    Check if the symbolic expression contains a specific item.
    """
    return in_(item, container)


def in_(item, container):
    """
    Check if the symbolic expression is in another iterable or symbolic expression.
    """
    return Comparator(item, 'in', container)
