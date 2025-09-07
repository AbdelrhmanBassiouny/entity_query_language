from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict

from typing_extensions import Any, Optional, List

from .enums import RDREdge
from .hashed_data import HashedValue
from .symbolic import SymbolicExpression, T, HasDomain, Variable


@dataclass(eq=False)
class Conclusion(SymbolicExpression[T], ABC):
    """
    Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).

    :ivar var: The variable being affected by the conclusion.
    :ivar value: The value or expression used by the conclusion.
    """
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

    def _reset_cache_(self) -> None:
        ...


@dataclass(eq=False)
class Set(Conclusion[T]):
    """Set the value of a variable in the current solution binding."""

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Dict[int, HashedValue]:
        if self._parent_._id_ not in sources:
            parent_value = next(iter(self._parent_._evaluate__(sources)))[self._parent_._id_]
        else:
            parent_value = sources[self._parent_._id_]
        parent_value.value = self.value
        return sources


@dataclass(eq=False)
class Add(Conclusion[T]):
    """Add a new value to the domain of a variable."""

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) -> Dict[int, HashedValue]:
        v = next(iter(self.value._evaluate__(sources)))[self.value._id_]
        self.var._domain_[v.id_] = v
        sources[self.var._id_] = v
        return sources