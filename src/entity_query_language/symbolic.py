from __future__ import annotations

import contextvars
from abc import abstractmethod, ABC
from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, ClassVar, Union, Generic, TypeVar
from typing_extensions import dataclass_transform, List, Tuple

from .failures import MultipleSolutionFound
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
class SymbolicExpression(Generic[T], ABC):
    _child_: Optional[SymbolicExpression] = field(init=False)
    _id_: int = field(init=False, repr=False, default=None)
    _node_: Node = field(init=False, default=None, repr=False)
    _id_expression_map_: ClassVar[Dict[int, SymbolicExpression]] = {}

    def __post_init__(self):
        self._id_ = id_generator(self)
        node_name = self._name_ + f"_{self._id_}"
        self._create_node_(node_name)
        if self._child_ is not None:
            self._update_child_()
        if self._id_ not in self._id_expression_map_:
            self._id_expression_map_[self._id_] = self

    def _update_child_(self):
        if self._child_._node_.parent is not None:
            child_cp = self._copy_child_expression_()
            self._child_ = child_cp
        self._child_._node_.parent = self._node_

    def _copy_child_expression_(self):
        child_cp = self._child_.__new__(self._child_.__class__)
        child_cp.__dict__.update(self._child_.__dict__)
        child_cp._create_node_(self._child_._node_.name + f"_{self._id_}")
        return child_cp

    def _create_node_(self, name: str):
        self._node_ = Node(name)
        self._node_._expression = self

    def _evaluate_(self, selected_vars: Iterable[HasDomain],
                   sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
        if isinstance(selected_vars, HasDomain):
            selected_vars = [selected_vars]
        seen_values = set()
        for v in self._evaluate__(sources):
            for var in selected_vars:
                if var._id_ not in v:
                    v[var._id_] = next(var._evaluate__(v))
            v = HashedIterable(values={var._id_: v[var._id_] for var in selected_vars})
            if v not in seen_values:
                seen_values.add(v)
                yield v

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
            return self._node_.parent._expression
        return None

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
        return self._node_.root._expression

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
        return [d._expression for d in self._node_.descendants]

    @property
    def _children_(self) -> List[SymbolicExpression]:
        return [c._expression for c in self._node_.children]

    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return OR(self, other)

    def __invert__(self):
        return Not(self)

    def __hash__(self):
        return hash(id(self))



@dataclass(eq=False)
class The(SymbolicExpression[T], Generic[T]):
    _child_: Entity[T]

    @property
    def _name_(self) -> str:
        return f"The({self._child_._name_})"

    def evaluate(self) -> T:
        return self._evaluate__()

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
class An(SymbolicExpression[T]):
    _child_: Entity[T]

    @property
    def _name_(self) -> str:
        return f"An({self._child_._name_})"

    def evaluate(self) -> Iterable[T]:
        yield from self._evaluate__()

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[T]:
        yield from self._child_._evaluate__(sources)



@dataclass(eq=False)
class SetOf(SymbolicExpression[T]):
    _child_: SymbolicExpression
    selected_variables_: Iterable[HasDomain]

    @property
    def _name_(self) -> str:
        return f"SetOf({self._child_._name_})"

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[Dict[HasDomain, Any]]:
        sol_gen = self._child_._evaluate_(self.selected_variables_, sources)
        for sol in sol_gen:
            yield {var: sol[var._id_].value for var in self.selected_variables_ if var._id_ in sol}


@dataclass(eq=False)
class Entity(SymbolicExpression[T]):
    _child_: SymbolicExpression
    selected_variable_: T

    @property
    def _name_(self) -> str:
        return f"Entity({self.selected_variable_.name_})"

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[T]:
        sol_gen = self._child_._evaluate_(self.selected_variable_)
        for sol in sol_gen:
            yield sol[self.selected_variable_._id_].value


@dataclass(eq=False)
class HasDomain(SymbolicExpression, ABC):
    _domain_: HashedIterable[Any] = field(default=None, init=False)
    _child_: HasDomain = field(init=False)

    def __post_init__(self):
        if self._domain_ is not None:
            self._domain_ = HashedIterable(self._domain_)
        super().__post_init__()

    def __iter__(self):
        yield from self._domain_

    def __getattr__(self, name):
        return Attribute(self, name)

    def __call__(self, *args, **kwargs):
        return Call(self, args, kwargs)

    def __eq__(self, other):
        return Comparator(self, '==', other)

    def __contains__(self, item):
        return Comparator(item, 'in', self)

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

    @property
    @lru_cache(maxsize=None)
    def _leaf_id_(self):
        return self._leaf_._id_

    @property
    @lru_cache(maxsize=None)
    def _leaf_(self) -> HasDomain:
        return list(self._leaves_)[0].value

    @property
    @lru_cache(maxsize=None)
    def _leaves_(self) -> HashedIterable[HasDomain]:
        if self._child_ is not None and hasattr(self._child_, '_leaves_'):
            return self._child_._leaves_
        else:
            value = HashedValue(value=self)
            return HashedIterable(values={value.id_: value})

    @property
    @lru_cache(maxsize=None)
    def _all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        child_leaves = []
        if self._child_ is not None and hasattr(self._child_, '_all_leaf_instances_'):
            child_leaves = self._child_._all_leaf_instances_
        return [self] + child_leaves

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class Variable(HasDomain):
    _cls_: Optional[Type] = field(default=None)
    _cls_kwargs_: Dict[str, Any] = field(default_factory=dict)
    _domain_: HashedIterable[Any] = field(default=None, kw_only=True)
    _child_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def _evaluate__(self, sources: Optional[HashedIterable[Any]] = None) -> Iterable[HashedValue]:
        """
        A variable does not need to evaluate anything by default.
        """
        sources = sources or HashedIterable()
        if self._id_ in sources:
            yield from (sources[self._id_],)
        else:
            if self._domain_ is None and self._cls_ is not None:
                def domain_gen():
                    cls_kwargs = {k: v._evaluate__(sources) if isinstance(v, HasDomain) else v for k, v in self._cls_kwargs_.items()}
                    symbolic_vars = []
                    for k, v in self._cls_kwargs_.items():
                        if isinstance(v, HasDomain):
                            symbolic_vars.append(v)
                    while True:
                        try:
                            instance = self._cls_(**{k: next(v).value if k in symbolic_vars else v.value for k, v in cls_kwargs.items()})
                            yield HashedValue(instance)
                        except StopIteration:
                            break

                yield from domain_gen()
            else:
                yield from self

    @property
    def _name_(self):
        return self._cls_.__name__

    @classmethod
    def _from_domain_(cls, iterable, clazz: Optional[Type] = None,
                      child: Optional[SymbolicExpression] = None) -> Variable:
        if not is_iterable(iterable):
            iterable = make_list(iterable)
        if not clazz:
            clazz = type(next((iter(iterable)), None))
        return Variable(clazz, _domain_=iterable, _child_=child)

    def __repr__(self):
        return (f"Symbolic({self._cls_.__name__}("
                f"{', '.join(f'{k}={v!r}' for k, v in self._cls_kwargs_.items())}))")


@dataclass(eq=False)
class DomainMapping(HasDomain, ABC):
    """
    A symbolic expression the maps the domain of symbolic variables.
    """
    _child_: HasDomain
    _invert_: bool = field(init=False, default=False)

    def _evaluate__(self, sources: Optional[HashedIterable] = None) \
            -> Iterable[Union[HashedIterable, HashedValue]]:
        child_val = self._child_._evaluate__(sources)
        if (self._conditions_root_ is self) or isinstance(self._parent_, LogicalOperator):
            for child_v in child_val:
                v = self._apply_mapping_(child_v)
                if (not self._invert_ and v.value) or (self._invert_ and not v.value):
                    yield HashedIterable(values={self._leaf_id_: self._leaf_._domain_[v.id_]})
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
    left_: SymbolicExpression
    operation_: str
    right_: SymbolicExpression
    _child_: SymbolicExpression = field(init=False, default=None)
    _invert__: bool = field(init=False, default=False)

    def __post_init__(self):
        if not isinstance(self.left_, SymbolicExpression):
            self.left_ = Variable._from_domain_([self.left_])
        if not isinstance(self.right_, SymbolicExpression):
            self.right_ = Variable._from_domain_([self.right_])
        super().__post_init__()
        for operand in [self.left_, self.right_]:
            operand._node_.parent = self._node_

    @property
    def _invert_(self):
        return self._invert__

    @_invert_.setter
    def _invert_(self, value):
        if value == self._invert__:
            return
        self._invert__ = value
        prev_operation = copy(self.operation_)
        match self.operation_:
            case "<":
                self.operation_ = ">=" if self._invert_ else self.operation_
            case ">":
                self.operation_ = "<=" if self._invert_ else self.operation_
            case "<=":
                self.operation_ = ">" if self._invert_ else self.operation_
            case ">=":
                self.operation_ = "<" if self._invert_ else self.operation_
            case "==":
                self.operation_ = "!=" if self._invert_ else self.operation_
            case "!=":
                self.operation_ = "==" if self._invert_ else self.operation_
            case "in":
                self.operation_ = "not in" if self._invert_ else self.operation_
        self._node_.name = self._node_.name.replace(prev_operation, self.operation_)

    @property
    def _name_(self):
        return self.operation_

    @property
    @lru_cache(maxsize=None)
    def _leaves_(self) -> HashedIterable[HasDomain]:
        return self.left_._leaves_ | self.right_._leaves_

    @property
    @lru_cache(maxsize=None)
    def _all_leaf_instances_(self) -> List[HasDomain]:
        """
        Get the leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
        """
        return self.left_._all_leaf_instances_ + self.right_._all_leaf_instances_


@dataclass(eq=False)
class Comparator(BinaryOperator):
    """
    A symbolic equality check that can be used to compare symbolic variables.
    """
    left_: HasDomain
    right_: HasDomain

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
        if self.right_._leaf_id_ not in sources:
            left_values = self.left_._evaluate__(sources)
            for left_value in left_values:
                right_values = self.right_._evaluate__(sources)
                for right_value in right_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

        else:
            right_values = self.right_._evaluate__(sources)
            for right_value in right_values:
                left_values = self.left_._evaluate__(sources)
                for left_value in left_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

    def check(self, left_value: HashedValue, right_value: HashedValue) -> Optional[HashedIterable]:
        satisfied = eval(f"left_value.value {self.operation_} right_value.value")
        if satisfied:
            left_leaf_value = self.left_._leaf_._domain_[left_value.id_]
            right_leaf_value = self.right_._leaf_._domain_[right_value.id_]
            return HashedIterable(values={self.left_._leaf_id_: left_leaf_value, self.right_._leaf_id_: right_leaf_value})
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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:

        # init an empty source if none is provided
        original_sources = sources or HashedIterable()
        sources = copy(original_sources)
        left_sources = copy(sources)
        right_values_leaf_ids = [leaf.id_ for leaf in self.right_._leaves_]
        seen_left_values = set()

        # constrain left values by available sources
        left_values = self.left_._evaluate__(left_sources)
        for left_value in left_values:

            values_for_right_leaves = {(k, left_value[k]) for k in right_values_leaf_ids if k in left_value}
            if seen_left_values and values_for_right_leaves.issubset(seen_left_values):
                continue
            else:
                seen_left_values.update(values_for_right_leaves)

            # update the sources
            sources = sources.union(left_value)

            # constrain right values by available sources
            right_values = self.right_._evaluate__(sources)

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

    def _evaluate__(self, sources: Optional[HashedIterable] = None) -> Iterable[HashedIterable]:
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
            operand_values = operand._evaluate__(operand_sources)

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
            return Variable(symbolic_cls, _cls_kwargs_=kwargs)
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
        operand = Variable._from_domain_(operand)
    if isinstance(operand, AND):
        operand = OR(Not(operand.left_), Not(operand.right_))
    elif isinstance(operand, OR):
        operand = AND(Not(operand.left_), Not(operand.right_))
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
