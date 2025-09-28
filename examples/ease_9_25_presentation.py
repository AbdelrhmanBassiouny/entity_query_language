from dataclasses import dataclass
from enum import Enum

from entity_query_language import symbol, a, predicate, the, entity, symbolic_mode, rule_mode


class Color(Enum):
    Orange = "orange"
    Blue = "blue"


@symbol
@dataclass
class Container:
    color: Color


@symbol
@dataclass
class Cook:
    ...

@predicate
def behind(obj1, obj2):
    return True

@predicate
def left_of(obj1, obj2):
    return True


Container(Color.Orange)
Cook()
with symbolic_mode():
    bottle = a(Container(color=Color.Orange))
    cook = a(Cook())
    bass_bottle = the(entity(bottle,
                             behind(bottle, cook),
                             left_of(bottle, cook)))

print(bass_bottle.evaluate())
bass_bottle._node_.visualize(label_max_chars_per_line=16, figsize=(25, 30),
                             font_size=35, node_size=10000)