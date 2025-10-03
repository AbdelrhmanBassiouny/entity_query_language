from typing import Type, Iterable, List, Set, Tuple, Union, Optional


def get_range_types(type_hint: Type) -> Iterable[Type]:
    origin = getattr(type_hint, '__origin__', None)
    args = getattr(type_hint, '__args__', [])
    if not origin:
        yield type_hint
    elif is_container_type(origin):
        range_type = args[0]
        if is_container_type(range_type):
            raise ValueError("Nested containers are not supported")
        else:
            yield range_type
    elif is_optional_type(origin, args):
        for arg in args:
            if arg is not type(None):
                yield arg
        yield type(None)


def is_container_type(type_origin: type) -> bool:
    return type_origin in [list, set, tuple, List, Set, Tuple]


def is_optional_type(type_origin: type, args: List[Type]) -> bool:
    return (type_origin is Optional) or (type_origin is Union and type(None) in args)
