from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import List

from typing_extensions import Callable, Optional, Any, dataclass_transform, Type, Tuple

from typing_extensions import ClassVar

from .cache_data import yield_class_values_from_cache, get_cache_keys_for_class_
from .enums import PredicateType, EQLMode
from .hashed_data import HashedValue
from .symbolic import T, SymbolicExpression, in_symbolic_mode, Variable, An, Entity, From, properties_to_expression_tree
from .utils import is_iterable


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


symbols_registry: List[Type] = []

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
    original_new = cls.__new__ if '__new__' in cls.__dict__ else object.__new__
    symbols_registry.append(cls)

    def symbolic_new(symbolic_cls, *args, **kwargs):
        domain, kwargs = update_domain_and_kwargs_from_args(symbolic_cls, *args, **kwargs)
        predicate_type = PredicateType.SubClassOfPredicate if issubclass(symbolic_cls, Predicate) else None
        # This mode is when we try to infer new instances of variables, this includes also evaluating predicates
        # because they also need to be inferred. So basically this mode is when there is no domain availabe and
        # we need to infer new values.
        if not domain and (in_symbolic_mode(EQLMode.Rule) or predicate_type):
            return Variable(symbolic_cls.__name__, symbolic_cls, _kwargs_=kwargs, _predicate_type_=predicate_type,
                            _is_indexed_=index_class_cache(symbolic_cls))
        else:
            # In this mode, we either have a domain through the `domain` provided here, or through the cache if
            # the domain is not provided. Then we filter this domain by the provided constraints on the variable
            # attributes given as keyword arguments.
            var, expression = extract_selected_variable_and_expression(symbolic_cls, domain, predicate_type,
                                                                       **kwargs)
            return An(Entity(expression, [var])) if expression else var

    def hybrid_new(symbolic_cls, *args, **kwargs):
        if in_symbolic_mode():
            return symbolic_new(symbolic_cls, *args, **kwargs)
        else:
            instance = instantiate_class_and_update_cache(symbolic_cls, original_new, *args, **kwargs)
            return instance

    cls.__new__ = hybrid_new
    return cls

cls_args = {}

def update_domain_and_kwargs_from_args(symbolic_cls: Type, *args, **kwargs):
    """
    Set the domain if provided as the first argument and update the kwargs with the remaining arguments.

    :param symbolic_cls: The constructed class.
    :param args: The positional arguments to the class constructor and optionally the domain.
    :param kwargs: The keyword arguments to the class constructor.
    :return: The domain and updated kwargs.
    """
    domain = None
    global cls_args
    if symbolic_cls not in cls_args:
        cls_args[symbolic_cls] = list(inspect.signature(symbolic_cls.__init__).parameters.keys())
    init_args = cls_args[symbolic_cls]
    for i, arg in enumerate(args):
        if isinstance(arg, From):
            domain = arg
            if i > 0:
                raise ValueError(f"First non-keyword-argument to {symbolic_cls.__name__} in symbolic mode should be"
                                 f" a domain using `From()`.")
        else:
            arg_name = init_args[i+1] # to skip `self`
            kwargs[arg_name] = arg
    return domain, kwargs



def extract_selected_variable_and_expression(symbolic_cls: Type, domain: Optional[From] = None,
                                             predicate_type: Optional[PredicateType] = None, **kwargs):
    """
    :param symbolic_cls: The constructed class.
    :param domain: The domain source for the values of the variable by.
    :param predicate_type: The predicate type.
    :param kwargs: The keyword arguments to the class constructor.
    :return: The selected variable and expression.
    """
    cache_keys = get_cache_keys_for_class_(Variable._cache_, symbolic_cls)
    if not domain and cache_keys:
        domain = From((v for a, v in yield_class_values_from_cache(Variable._cache_, symbolic_cls, from_index=False,
                                                                   cache_keys=cache_keys)))
    elif domain and is_iterable(domain.domain):
            domain.domain = filter(lambda v: isinstance(v, symbolic_cls), domain.domain)

    var = Variable(symbolic_cls.__name__, symbolic_cls, _domain_source_=domain, _predicate_type_=predicate_type,
                   _is_indexed_=index_class_cache(symbolic_cls))

    expression, _ = properties_to_expression_tree(var, kwargs)

    return var, expression



def instantiate_class_and_update_cache(symbolic_cls: Type, original_new: Callable, *args, **kwargs):
    """
    :param symbolic_cls: The constructed class.
    :param original_new: The original class __new__ method.
    :param args: The positional arguments to the class constructor.
    :param kwargs: The keyword arguments to the class constructor.
    :return: The instantiated class.
    """
    instance = original_new(symbolic_cls)
    index = index_class_cache(symbolic_cls)
    if index:
        if symbolic_cls not in cls_args:
            cls_args[symbolic_cls] = list(inspect.signature(symbolic_cls.__init__).parameters.keys())
        kwargs = {f: HashedValue(getattr(instance, f)) for f in cls_args[symbolic_cls][1:]}
        if symbolic_cls not in Variable._cache_ or not Variable._cache_[symbolic_cls].keys:
            Variable._cache_[symbolic_cls].keys = kwargs.keys()
    else:
        kwargs = {}
    Variable._cache_[symbolic_cls].insert(kwargs, HashedValue(instance), index=index)
    return instance

def index_class_cache(symbolic_cls: Type) -> bool:
    """
    Determine whether the class cache should be indexed.
    """
    return False
    # return issubclass(symbolic_cls, Predicate) and symbolic_cls.is_expensive


@symbol
@dataclass(eq=False)
class Predicate(ABC):
    """
    The super predicate class that represents a filtration operation.
    """
    is_expensive: ClassVar[bool] = False

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
    types_: Type
    is_expensive: ClassVar[bool] = False

    def __call__(self) -> bool:
        return isinstance(self.variable, self.types_)


@dataclass(eq=False)
class HasTypes(HasType):
    types_: Tuple[Type, ...]
