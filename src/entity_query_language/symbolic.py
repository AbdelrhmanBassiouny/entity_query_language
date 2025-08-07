from __future__ import annotations

import contextvars
import itertools
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from functools import lru_cache

from anytree import Node
from typing_extensions import Iterable, Any, Optional, Type, Dict, Set, ClassVar, Callable, Union
from typing_extensions import dataclass_transform, List, Tuple

from .utils import is_iterable, filter_data
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


@dataclass
class HashedValue:
    value: Any
    id_: Optional[int] = field(default=None)

    def __hash__(self):
        if self.id_ is None and hasattr(self.value, "id_"):
            return hash(self.value.id_)
        elif self.id_ is not None:
            return hash(self.id_)
        else:
            return hash(id(self.value))


@dataclass
class HashedIterable:
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """
    iterable: Iterable[Any] = field(default_factory=list)
    values: Dict[int, HashedValue] = field(default_factory=dict)

    def __post_init__(self):
        if self.iterable and not isinstance(self.iterable, HashedIterable):
            self.iterable = (HashedValue(id_=k, value=v) if not isinstance(v, HashedValue) else v
                             for k, v in enumerate(self.iterable))

    @property
    def ids(self) -> Set[int]:
        """
        Get the ids of the hashed values.

        :return: A set of ids of the hashed values.
        """
        return set(self.values.keys())

    def map(self, func: Callable[[Any], Any]):
        """
        Apply a function to each value in the HashedIterable and return a new HashedIterable.

        :param func: The function to apply to each value.
        :return: A new HashedIterable with the transformed values.
        """
        return HashedIterable(map(lambda v: HashedValue(id_=v.id_, value=func(v.value)), self.iterable))

    def filter(self, selected_ids: Iterable[int], ids_are_indices: bool = False):
        """
        Filter the HashedIterable based on a set of selected ids.

        :param selected_ids: An iterable of ids to keep in the HashedIterable.
        :param ids_are_indices: Whether the ids are indices into the iterable. If True, the ids are used as
         indices into the iterable. If False, the ids are used as the hash ids of the HashedValues.
        :return: A new HashedIterable containing only the items with the specified ids.
        """
        if ids_are_indices:
            to_filter, self.iterable = itertools.tee(self, 2)
            self.iterable = filter_data(to_filter, selected_ids)
        else:
            it1, it2 = itertools.tee(self.iterable)
            self.iterable = (v for v in it1 if v.id_ in selected_ids)
        self.values = {}
        return HashedIterable(self.iterable)

    def update(self, values: HashedIterable):
        """
        Update the hashed values with another HashedIterable.

        :param values: The HashedIterable to update with.
        """
        for id_, v in values.values.items():
            self.add(v)

    def add(self, value: HashedValue):
        """
        Add a HashedValue to the hashed values.

        :param value: The HashedValue to add.
        """
        # if value.id_ not in self.values:
        self.values[value.id_] = value
        # else:
        #     raise ValueError(f"Value with id {value.id_} already exists in the hashed values.")

    def get_unique_values(self) -> Iterable[HashedValue]:
        seen_values = set()
        for v in self.iterable:
            if v not in seen_values:
                seen_values.add(v)
                yield v

    def union(self, other: HashedIterable) -> HashedIterable:
        """
        Create a union of two HashedIterables.

        :param other: The other HashedIterable to union with.
        :return: A new HashedIterable containing the union of both.
        """

        def union():
            seen_values = set()
            for v in self.iterable:
                if v not in seen_values:
                    seen_values.add(v)
                    yield v
            for v in other.iterable:
                if v not in seen_values:
                    seen_values.add(v)
                    yield v

        return HashedIterable(union())

    def intersection(self, other: HashedIterable) -> HashedIterable:
        common_keys = self.values.keys() & other.values.keys()
        return HashedIterable({k: self.values[k] for k in common_keys})

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        yield from self.values.values()
        for v in self.iterable:
            self.values[v.id_] = v
            yield v

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
        iterable_copy, self.iterable = itertools.tee(self.iterable, 2)
        return HashedIterable(values=self.values.copy(), iterable=iterable_copy)


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

    @abstractmethod
    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
        """
        Evaluate the symbolic expression and set the operands indices.
        This method should be implemented by subclasses.
        """
        pass

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
        if name.startswith('_') or name in ['leaves_', 'child_', 'all_leaf_instances_']:
            raise AttributeError(name)
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
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

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
    domain_: HashedIterable = field(default=None, init=False)

    def __post_init__(self):
        if self.domain_ is not None:
            self.domain_ = HashedIterable(self.domain_)
        super().__post_init__()

    def __iter__(self):
        yield from self.domain_

    def constrain_(self, ids: Iterable[int]):
        if self.child_ is not None and isinstance(self.child_, HasDomain):
            self.child_.constrain_(ids)
        elif self.child_ is None:
            self.processing_domain_.extend(self.domain_.filter(ids))

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
    def leaves_(self) -> Set[HashedValue]:
        if self.child_ is not None and hasattr(self.child_, 'leaves_'):
            return self.child_.leaves_
        else:
            return {HashedValue(value=self)}

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
    domain_: HashedIterable = field(default=None, kw_only=True)
    child_: Optional[SymbolicExpression] = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.domain_ is None and self.cls is not None:
            def domain_gen():
                cls_kwarg_gen = {k: iter(v) if is_iterable(v) else v for k, v in self.cls_kwargs_.items()}
                while True:
                    try:
                        yield self.cls_(**{k: next(kwarg_gen) if is_iterable(kwarg_gen) else kwarg_gen
                                           for k, kwarg_gen in cls_kwarg_gen.items()})
                    except StopIteration:
                        break

            self.domain_: HashedIterable = HashedIterable(domain_gen())

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[HashedValue]:
        """
        A variable does not need to evaluate anything by default.
        """
        sources = sources or {}
        if self.id_ in sources:
            yield from (sources.get(self.id_),)
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

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[HashedValue]:
        child_val = self.child_.evaluate_(sources)
        if self.root_ is self:
            indices = [v.id_ for v in self if v.value]
            condition = lambda v: v.id_ in indices
            yield from ({self.leaf_id_: self.leaf_.domain_[v.id_]} for v in child_val if condition(v))
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
    operands_values_: Dict[int, HashedValue] = field(default_factory=lambda: defaultdict(HashedIterable), init=False)

    def constrain_(self):
        """
        Constrain the symbolic expression based on the indices.
        This method should be implemented by subclasses.
        """
        for operand_id, value in self.operands_values_.items():
            self.id_expression_map_[operand_id].constrain_([value.id_])

    @property
    @abstractmethod
    def leaves_(self) -> Set[HashedValue]:
        """
        :return: Set of leaves of symbolic expressions, these are the variables that will have their domains constrained.
        """
        ...

    @property
    @abstractmethod
    def all_leaf_instances_(self) -> List[HasDomain]:
        """
        :return: List of all leaf instances of the symbolic expression.
        This is useful for accessing the leaves of the symbolic expression tree.
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
    def leaves_(self) -> Set[HashedValue]:
        return self.operand_.leaves_

    @property
    @lru_cache(maxsize=None)
    def all_leaf_instances_(self) -> List[HasDomain]:
        return self.operand_.all_leaf_instances_


@dataclass(eq=False)
class Not(UnaryOperator):
    """
    A symbolic NOT operation that can be used to negate symbolic expressions.
    """

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None):
        def operator_yield():
            yield from (id_ for id_, value in self.operand_ if not value)

        operand_leaf = self.operand_.leaves_.pop().value
        operand_leaf.domain_.filter(operator_yield())


@dataclass(eq=False)
class BinaryOperator(ConstrainingOperator, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """
    left_: HasDomain
    operation_: str
    right_: HasDomain
    child_: SymbolicExpression = field(init=False, default=None)

    def __post_init__(self):
        if not isinstance(self.left_, SymbolicExpression):
            self.left_ = Variable.from_domain_(self.left_)
        if not isinstance(self.right_, SymbolicExpression):
            self.right_ = Variable.from_domain_(self.right_)
        super().__post_init__()
        for operand in [self.left_, self.right_]:
            operand.node_.parent = self.node_

    @property
    def name_(self):
        return self.operation_

    @property
    @lru_cache(maxsize=None)
    def leaves_(self) -> Set[HashedValue]:
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

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:
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
        sources = sources or {}
        if self.right_.leaf_id_ not in sources:
            left_values = self.left_.evaluate_(sources)
            for left_value in left_values:
                right_values = self.right_.evaluate_(sources)
                for right_value in right_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

        else:
            right_values = self.right_.evaluate_(sources)
            for right_value in right_values:
                left_values = self.left_.evaluate_(sources)
                for left_value in left_values:
                    res = self.check(left_value, right_value)
                    if res:
                        yield res

    def check(self, left_value: HashedValue, right_value: HashedValue) -> Optional[Dict[int, HashedValue]]:
        satisfied = eval(f"left_value.value {self.operation_} right_value.value")
        if satisfied:
            left_leaf_value = self.left_.leaf_.domain_[left_value.id_]
            right_leaf_value = self.right_.leaf_.domain_[right_value.id_]
            return {self.left_.leaf_id_: left_leaf_value, self.right_.leaf_id_: right_leaf_value}
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

    @staticmethod
    def check(node: HasDomain, value: Union[HashedValue, Dict[int, HashedValue]]) -> Optional[Dict[int, HashedValue]]:
        if not isinstance(node, ConstrainingOperator):
            if value.value:
                return {node.leaf_id_: node.leaf_.domain_[value.id_]}
            else:
                return None
        else:
            return value


@dataclass(eq=False)
class And(LogicalOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None) -> Iterable[Dict[int, HashedValue]]:

        # init an empty source if none is provided
        original_sources = sources or {}
        sources = copy(original_sources)
        left_sources = copy(sources)

        # constrain left values by available sources
        left_values = self.left_.evaluate_(left_sources)
        for left_value in left_values:
            # Check left value, if result is False, continue to next left value.
            left_value = self.check(self.left_, left_value)
            if not left_value:
                continue

            # update the sources
            sources.update(left_value)
            # constrain right values by available sources
            right_values = self.right_.evaluate_(sources)

            # For the found left value, find all right values,
            # and yield the (left, right) results found.
            for right_value in right_values:
                right_value = self.check(self.right_, right_value)
                if not right_value:
                    continue

                sources.update(right_value)

                yield sources
            sources = copy(original_sources)


@dataclass(eq=False)
class Or(LogicalOperator):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """

    def __post_init__(self):
        super().__post_init__()
        # Find common leaves between operands and split them into separate leaves, each connected to a separate operand.
        # This is necessary to ensure that the leaves of the OR operator are not shared between operands, which would
        # make the evaluation of each operand affect the others. Instead, we want each operand to have a copy of the
        # leaves, so that they can be evaluated independently. The leaves here are the symbolic variables that will be
        # constrained by the OR operator.
        all_leaves = [operand.all_leaf_instances_ for operand in [self.left_, self.right_]]
        unique_leaves = [operand.leaves_ for operand in [self.left_, self.right_]]
        shared_leaves = set.intersection(*unique_leaves)
        for leaf_hashed_value in shared_leaves:
            leaf = leaf_hashed_value.value
            first_occurrence = True
            for operand_leaves in all_leaves:
                if leaf.id_ not in {l.id_ for l in operand_leaves}:
                    continue
                if first_occurrence:
                    first_occurrence = False
                    continue
                leaf_instances = [l for l in operand_leaves if l.id_ == leaf.id_]
                leaf_instances[0].domain_ = copy(leaf.domain_)
                for leaf_instance in leaf_instances[1:]:
                    leaf_instance.domain_ = leaf_instances[0].domain_

    def evaluate_(self, sources: Optional[Dict[int, HashedValue]] = None):
        """
        Constrain the symbolic expression based on the indices of the operands.
        This method overrides the base class method to handle OR logic.
        """
        # init an empty source if none is provided
        original_sources = sources or {}

        # constrain left values by available sources
        for operand in [self.left_, self.right_]:
            operand_sources = copy(original_sources)
            operand_values = operand.evaluate_(operand_sources)

            for operand_value in operand_values:
                # Check operand value, if result is False, continue to next operand value.
                operand_value = self.check(operand, operand_value)
                if not operand_value:
                    continue
                operand_sources.update(operand_value)
                yield operand_sources
                operand_sources = copy(original_sources)


@dataclass_transform()
def symbolic(cls):
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            return Variable(symbolic_cls, cls_kwargs_=kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls


def and_(*conditions):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """
    return And(list(conditions))


def or_(*conditions):
    """
    A symbolic OR operation that can be used to combine multiple symbolic expressions.
    """
    return Or(list(conditions))


def in_(item, container):
    """
    Check if the symbolic expression is in another iterable or symbolic expression.
    """
    return Comparator(item, 'in', container)


def contains(container, item):
    """
    Check if the symbolic expression contains a specific item.
    """
    return in_(item, container)
