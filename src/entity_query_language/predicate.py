from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from typing_extensions import Callable, Any, Tuple, Dict, Optional, List, Iterable, Union

from .hashed_data import T
from .hashed_data import HashedIterable, HashedValue
from .symbolic import SymbolicExpression, HasDomain, Variable, LogicalOperator
from .utils import generate_combinations


@dataclass(eq=False)
class Predicate(SymbolicExpression[T]):
    """
    A symbolic expression that represents a predicate function applied to symbolic variables.
    """
    _function_: Callable[[Any], Any]
    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    _child_vars_: Dict[str, HasDomain] = field(default_factory=dict)
    _invert_: bool = field(init=False, default=False)

    def __post_init__(self):
        if type(self) is Predicate:
            self._child_ = None
        function_arg_names = [pname for pname, p in inspect.signature(self._function_).parameters.items()
                              if p.default == inspect.Parameter.empty]
        self._kwargs_.update(dict(zip(function_arg_names, self._args_)))
        if not all(name in self._kwargs_ for name in function_arg_names):
            raise ValueError(f"The number of arguments of the predicate function {self._function_.__name__} "
                             f"does not match the number of provided arguments.")
        if self._kwargs_:
            for k, v in self._kwargs_.items():
                self._update_child_vars_(v, name=k)
        super().__post_init__()
        self._update_children_(*self._child_vars_.values())

    def _update_child_vars_(self, source: Any, name: Optional[str] = None):
        if not isinstance(source, HasDomain):
            source = Variable._from_domain_(source, name=name)
        self._child_vars_[name] = source

    @property
    def _name_(self):
        return f"{self._function_.__name__}({','.join(f'{k}={v._name_}' for k, v in self._child_vars_.items())})"

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        variables = []
        for k, v in self._child_vars_.items():
            variables.extend(v._all_variable_instances_)
        return variables

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None) \
            -> Iterable[Dict[int, HashedValue]]:
        kwargs_generators = {k: v._evaluate__(sources) for k, v in self._child_vars_.items()}
        for kwargs in generate_combinations(kwargs_generators):
            function_output = self._function_(**{k: v[self._child_vars_[k]._id_].value for k, v in kwargs.items()})
            if (not self._invert_ and function_output) or (self._invert_ and not function_output):
                self._is_false_ = False
            else:
                self._is_false_ = True
            if self._yield_when_false_ or not self._is_false_:
                values = {}
                for k, v in kwargs.items():
                    values.update(v)
                values[self._id_] = HashedValue(function_output)
                yield values

    def _reset_cache_(self) -> None:
        ...
