from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import wraps

from typing_extensions import Callable, Any, Tuple, Optional, dataclass_transform, Type

from .enums import EQLMode, PredicateType
from .hashed_data import HashedValue, HashedIterable
from .hashed_data import T
from .symbolic import SymbolicExpression, Variable, in_symbolic_mode, Entity, An, chained_logic, AND


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
            domain = kwargs.pop('domain', HashedIterable())
            predicate_type = PredicateType.SubClassOfPredicate if issubclass(symbolic_cls, Predicate) else None
            # This mode is when we try to infer new instances of variables, this includes also evaluating predicates
            # because they also need to be inferred. So basically this mode is when there is no domain availabe and
            # we need to infer new values.
            if not domain and (in_symbolic_mode(EQLMode.Rule) or predicate_type):
                return Variable(symbolic_cls.__name__, symbolic_cls, _kwargs_=kwargs, _domain_=domain,
                                _predicate_type_=predicate_type)
            else:
                # In this mode, we either have a domain through the `domain` provided here, or through the cache if
                # the domain is not provided. Then we filter this domain by the provided constraints on the variable
                # attributes given as keyword arguments.
                var = Variable(symbolic_cls.__name__, symbolic_cls, _domain_=domain, _predicate_type_=predicate_type)
                if kwargs:
                    conditions = [getattr(var, k) == v for k, v in kwargs.items()]
                    if len(conditions) == 1:
                        expression = conditions[0]
                    else:
                        expression = chained_logic(AND, *conditions)
                    return An(Entity([var], expression))
                else:
                    return var
        else:
            instance = orig_new(symbolic_cls)
            instance.__init__(*args, **kwargs)
            kwargs = {f.name: HashedValue(getattr(instance, f.name)) for f in fields(instance) if f.init}
            if symbolic_cls not in Variable._cache_ or not Variable._cache_[symbolic_cls].keys:
                Variable._cache_[symbolic_cls].keys = list(kwargs.keys())
            Variable._cache_[symbolic_cls].insert(kwargs, HashedValue(instance))
            return instance

    cls.__new__ = symbolic_new
    return cls


@symbol
@dataclass(eq=False)
class Predicate(ABC):
    """
    The super predicate class that represents a filtration operation.
    """

    @abstractmethod
    def __call__(self) -> Any:
        """
        Evaluate the predicate with the current arguments and return the results.
        This method should be implemented by subclasses.
        """
        ...


@dataclass(eq=False)
class HasType(Predicate):
    variable: Any
    types_: Tuple[Type, ...]

    def __call__(self) -> bool:
        return isinstance(self.variable, self.types_)
