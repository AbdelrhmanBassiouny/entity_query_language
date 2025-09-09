from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import lru_cache, wraps

from typing_extensions import Callable, Any, Tuple, Dict, Optional, List, Iterable, ClassVar, dataclass_transform, Type

from .cache_data import IndexedCache
from .enums import InferMode
from .hashed_data import HashedValue
from .hashed_data import T
from .symbolic import SymbolicExpression, HasDomain, Variable, in_symbolic_mode
from .utils import generate_combinations


def predicate(function: Callable[..., T]) -> Callable[..., SymbolicExpression[T]]:
    """
    Function decorator that constructs a symbolic expression representing the function call
     when inside a symbolic_rule context.

    When symbolic mode is active, calling the method returns a Call instance which is a SymbolicExpression bound to
    representing the method call that is not evaluated until the evaluate() method is called on the query/rule.

    :param function: The function to decorate.
    :return: The decorated function.
    """

    @wraps(function)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        if in_symbolic_mode():
            return SymbolicPredicate(function, _args_=args, _kwargs_=kwargs)
        return function(*args, **kwargs)

    return wrapper


@dataclass_transform()
def symbol(cls):
    """
    Class decorator that makes a class construct symbolic Variables when inside
    a symbolic_rule context.

    When symbolic mode is active, calling the class returns a Variable bound to
    either a provided domain or to deferred keyword domain sources.

    :param cls: The class to decorate.
    :return: The same class with a patched ``__new__``.
    """
    orig_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__

    def symbolic_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            if issubclass(symbolic_cls, Predicate):
                instance = orig_new(symbolic_cls)
                instance.__init__(*args, **kwargs)
                return instance.symbolic_instance
            if len(args) > 0:
                return Variable(args[0], symbolic_cls, _domain_=args[1])
            return Variable(symbolic_cls.__name__, symbolic_cls, _cls_kwargs_=kwargs)
        return orig_new(symbolic_cls)

    cls.__new__ = symbolic_new
    return cls

PredicateCache = Dict[str, IndexedCache]

@symbol
@dataclass(eq=False)
class Predicate(ABC):
    """
    The super predicate class that represents a relation. This class manages an in memory graph of relations that is
    used to cache results of functions/predicates and allow retrieval from the graph instead of re-evaluation of the
    predicate each time. This also means that cache invalidation mechanisms are needed to remove relations that no
    longer hold which would have caused incorrect query results.

    This class changes the behavior of functions in symbolic mode to return a SymbolicPredicate instead of evaluating
    the function.
    """
    infer_mode: InferMode = field(default=InferMode.Auto, init=False)
    """
    The inference mode to use, can be Always, Never, and Auto (requires implementation of _should_infer method that
    decides when to infer and when to retrieve the results).
    """
    inferred_once: ClassVar[bool] = False
    """
    If inference has been performed at least once.
    """
    symbolic_instance: SymbolicPredicate = field(init=False)
    """
    The symbolic instance of the predicate, this is used to construct the symbolic expression from the predicate.
    """
    cache: ClassVar[PredicateCache] = defaultdict(IndexedCache)
    """
    A mapping from predicate name to a dict mapping edge individual id to its arguments.
    """

    def __post_init__(self):
        if in_symbolic_mode():
            self.symbolic_instance = SymbolicPredicate(self, _kwargs_=self.predicate_kwargs)
        if self.edge_name() not in self.cache:
            self.cache[self.edge_name()].keys = list(self.predicate_kwargs.keys())

    def __call__(self) -> Any:
        """
        This method wraps the behavior of the predicate by deciding whether to retrieve or to infer and also adds
        new relations to the graph through the infer() method.
        """
        if self.should_infer:
            result = self.infer()
            self.__class__.inferred_once = True
        else:
            result = self.retrieve()
        return result

    def retrieve(self) -> Optional[Any]:
        """
        Retrieve the results of the predicate from the graph.
        """
        for kwargs, result in self.cache[self.edge_name()].retrieve(self.predicate_kwargs):
            return result
        return None

    def infer(self) -> Any:
        """
        Evaluate the predicate and infer new relations and add them to the graph.
        """
        result = self._infer()
        self.cache[self.edge_name()].insert(self.predicate_kwargs, result)
        return result

    @property
    def should_infer(self) -> bool:
        """
        Determine if the predicate relations should be inferred or just retrieve current relations.
        """
        match self.infer_mode:
            case InferMode.Always:
                return True
            case InferMode.Never:
                return False
            case InferMode.Auto:
                return self._should_infer()
            case _:
                raise ValueError(f"Invalid infer mode: {self.infer_mode}")

    @classmethod
    def clear_cache(cls):
        """
        Clear the cache of the predicate.
        """
        cls.cache[cls.edge_name()].clear()

    @property
    def predicate_kwargs(self) -> Dict[str, Any]:
        """
        The keyword arguments to pass to the predicate.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def _should_infer(self) -> bool:
        """
        Predicate specific reasoning on when to infer relations.
        """
        return not self.cache[self.edge_name()].check(self.predicate_kwargs)

    @classmethod
    def edge_name(cls):
        return cls.__name__

    @abstractmethod
    def _infer(self) -> Any:
        """
        Evaluate the predicate with the given arguments and return the results.
        This method should be implemented by subclasses.
        """
        ...


@dataclass(eq=False)
class SymbolicPredicate(SymbolicExpression[T]):
    """
    A symbolic expression that represents a predicate function applied to symbolic variables.
    """
    _function_: Callable[[Any], Any]
    _args_: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs_: Dict[str, Any] = field(default_factory=dict)
    _child_vars_: Dict[str, HasDomain] = field(default_factory=dict)
    _invert_: bool = field(init=False, default=False)

    def __post_init__(self):
        if type(self) is SymbolicPredicate:
            self._child_ = None
        if not self.is_function_a_predicate_instance:
            function_arg_names = [pname for pname, p in inspect.signature(self._function_).parameters.items()
                                  if p.default == inspect.Parameter.empty]
            self._kwargs_.update(dict(zip(function_arg_names, self._args_)))
            if not all(name in self._kwargs_ for name in function_arg_names if name not in ['args', 'kwargs']):
                raise ValueError(f"The number of arguments of the predicate function {self._name_} "
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
        args_kwargs_str = f"({','.join(f'{k}={v._name_}' for k, v in self._child_vars_.items())})"
        if self.is_function_a_predicate_instance:
            return self._function_.__class__.__name__ + args_kwargs_str
        return f"{self._function_.__name__}{args_kwargs_str}"

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
            if self.is_function_a_predicate_instance:
                for k, v in kwargs.items():
                    setattr(self._function_, k, v[self._child_vars_[k]._id_].value)
                function_output = self._function_()
            else:
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

    @property
    def is_function_a_predicate_instance(self):
        return isinstance(self._function_, Predicate)

    def _reset_cache_(self) -> None:
        ...
