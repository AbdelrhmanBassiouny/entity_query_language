from dataclasses import dataclass, field
from typing import List

from entity_query_language import a, set_of, symbolic_mode, let, flatten
from entity_query_language.predicate import symbol

# Minimal dataset for the example
@symbol
@dataclass
class Body:
    name: str

@symbol
@dataclass
class Handle(Body):
    ...

@symbol
@dataclass
class Container(Body):
    ...

@symbol
@dataclass
class View:
    world: object = field(default=None, repr=False, kw_only=True)

@symbol
@dataclass
class Drawer(View):
    handle: Handle
    container: Container

@symbol
@dataclass
class World:
    bodies: List[Body] = field(default_factory=list)
    views: List[View] = field(default_factory=list)

# Build a small world
world = World()
container1 = Container(name="Container1")
container3 = Container(name="Container3")
handle1 = Handle(name="Handle1")
handle3 = Handle(name="Handle3")
world.bodies.extend([container1, container3, handle1, handle3])

# Two drawers
drawer1 = Drawer(handle=handle1, container=container1)
drawer2 = Drawer(handle=handle3, container=container3)

# A simple view-like class with an iterable attribute
class CabinetLike(View):
    def __init__(self, drawers):
        super().__init__()
        self.drawers = list(drawers)

cabinet = CabinetLike([drawer1, drawer2])
world.views = [cabinet]

with symbolic_mode():
    views = let(type_=View, domain=world.views)
    drawers = flatten(views.drawers)  # <-- UNNEST-like flatten
    query = a(set_of([views, drawers]))

rows = list(query.evaluate())
# rows is a list of UnificationDict, each containing both the parent view and one flattened drawer
assert len(rows) == 2
assert {r[drawers].handle.name for r in rows} == {"Handle1", "Handle3"}
assert all(r[views] is cabinet for r in rows)