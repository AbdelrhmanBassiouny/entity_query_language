from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from entity_query_language import symbol, a, predicate, the, entity, symbolic_mode, an


class Color(Enum):
    Orange = "orange"
    Black = "black"
    Grey = "grey"


@symbol
@dataclass
class PhysicalObject:
    color: Optional[Color] = None


@dataclass
class Container(PhysicalObject):
    ...


@dataclass
class Oven(PhysicalObject):
    ...


@symbol
@dataclass
class Hand:
    ...

@dataclass
class LeftHand(Hand):
    ...

@dataclass
class RightHand(Hand):
    ...

@symbol
@dataclass
class Cook:
    right_hand: PhysicalObject = field(default_factory=RightHand)
    left_hand: PhysicalObject = field(default_factory=LeftHand)

@dataclass
class CookingPot(PhysicalObject):
    ...

@dataclass
class WoodenSpoon(PhysicalObject):
    ...

@predicate
def behind(obj1, obj2):
    return True

@predicate
def left_of(obj1, obj2):
    return True

@predicate
def infront_of(obj1, obj2):
    return True

@predicate
def on(obj1, obj2):
    return True

@predicate
def above(obj1, obj2):
    return True

@predicate
def in_contact(obj1, obj2):
    ...


Container(Color.Orange)
Cook()
CookingPot(Color.Grey)
CookingPot(Color.Black)
WoodenSpoon()
Oven()
Oven()

with symbolic_mode():
    bottle = a(Container(color=Color.Orange))
    cook = a(Cook())
    bass_bottle = the(entity(bottle,
                             behind(bottle, cook),
                             left_of(bottle, cook)))

# print(bass_bottle.evaluate())
# bass_bottle._node_.visualize(label_max_chars_per_line=16, figsize=(30, 25),
#                              font_size=35, node_size=5000)

with symbolic_mode():
    involved_pot = the(pot:=CookingPot(color=Color.Grey),
                       infront_of(pot, cook),
                       on(pot, an(Oven())),
                       in_contact(pot, the(WoodenSpoon())))

# print(involved_pot.evaluate())
# involved_pot._node_.visualize(label_max_chars_per_line=16, figsize=(30, 25),
#                              font_size=35, node_size=5000)


with symbolic_mode():
    held_object = the(container := Container(),
                      in_contact(container, cook.left_hand)
                      & above(container, a(CookingPot())))

# print(held_object.evaluate())
held_object._node_.visualize(label_max_chars_per_line=16, figsize=(20, 20),
                             font_size=35, node_size=7000, spacing_y=1, spacing_x=1)