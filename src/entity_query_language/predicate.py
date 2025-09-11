from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import lru_cache, wraps

from typing_extensions import Callable, Any, Tuple, Dict, Optional, List, Iterable, ClassVar, dataclass_transform, Type

from .cache_data import IndexedCache
from .enums import InferMode, EQLMode, PredicateType
from .failures import ValueNotFoundInCache, MoreThanOneCacheEntryMatched
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
            function_arg_names = [pname for pname, p in inspect.signature(function).parameters.items()
                                  if p.default == inspect.Parameter.empty]
            kwargs.update(dict(zip(function_arg_names, args)))
            return Variable(function.__name__, function, _kwargs_=kwargs,
                            _predicate_type_=PredicateType.DecoratedMethod)
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
            if len(args) > 0:
                return Variable(args[0], symbolic_cls, _domain_=args[1])
            predicate_type = PredicateType.SubClassOfPredicate if issubclass(symbolic_cls, Predicate) else None
            return Variable(symbolic_cls.__name__, symbolic_cls, _kwargs_=kwargs, _predicate_type_=predicate_type)
        else:
            instance = orig_new(symbolic_cls)
            instance.__init__(*args, **kwargs)
            kwargs = {f.name: HashedValue(getattr(instance, f.name)) for f in fields(instance) if f.init}
            if symbolic_cls not in Variable._cache_ or not Variable._cache_[symbolic_cls].keys :
                Variable._cache_[symbolic_cls].keys = list(kwargs.keys())
            Variable._cache_[symbolic_cls].insert(kwargs, HashedValue(instance))
            return instance

    cls.__new__ = symbolic_new
    return cls

PredicateCache = Dict[Type, IndexedCache]

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
    all_predicates_cache: ClassVar[PredicateCache] = defaultdict(IndexedCache)
    """
    A mapping from predicate name to a dict mapping edge individual id to its arguments.
    """

    def __post_init__(self):
        if type(self) not in self.all_predicates_cache:
            self.cache.keys = list(self.predicate_kwargs.keys())

    def __call__(self) -> Any:
        """
        This method wraps the behavior of the predicate by deciding whether to retrieve or to infer and also adds
        new relations to the graph through the infer() method.
        """
        if self.should_infer:
            result = self.infer()
            self.__class__.inferred_once = True
        else:
            result = self.retrieve_one()[1]
        return result

    def retrieve_one(self) -> Tuple[Dict, Any]:
        """
        Retrieve the result of the predicate from the graph, should only return one result.

        :return: The results of the predicate.
        :raises ValueError: If no results are found.
        :raises MoreThanOneCacheEntryMatched: If more than one result is found.
        """
        val_gen = self.retrieve()
        result = None
        try:
            result = next(val_gen)
            result2 = next(val_gen)
            raise MoreThanOneCacheEntryMatched(self, self.predicate_kwargs, [result, result2])
        except StopIteration:
            if result is not None:
                return result
            raise ValueNotFoundInCache(self, self.predicate_kwargs)

    def retrieve(self) -> Iterable[Tuple[Dict, Any]]:
        """
        Retrieve the results of the predicate from the graph.
        """
        yield from self.cache.retrieve(self.predicate_kwargs)

    def infer(self) -> Any:
        """
        Evaluate the predicate and infer new relations and add them to the graph.
        """
        result = self._infer()
        self.cache.insert(self.predicate_kwargs, result)
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
        cls.all_predicates_cache[cls].clear()

    @classmethod
    def clear_all_predicate_caches(cls):
        """
        Clear the caches of all predicates.
        """
        for cache in cls.all_predicates_cache.values():
            cache.clear()

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
        return not self.cache.check(self.predicate_kwargs)

    @property
    def cache(self) -> IndexedCache:
        return self.all_predicates_cache[type(self)]

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
