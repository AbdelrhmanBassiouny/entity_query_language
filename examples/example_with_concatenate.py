from dataclasses import dataclass, field
from typing import List

from entity_query_language import symbolic_mode, let, concatenate, not_, in_, entity, an, From
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


# A simple view-like class with an iterable attribute `drawers`
class CabinetLike(View):
    def __init__(self, drawers):
        super().__init__()
        self.drawers = list(drawers)


cabinet = CabinetLike([drawer1, drawer2])
world.views = [cabinet]

# Example 1: Use concatenate to collect all drawers into a single iterable domain
with symbolic_mode():
    views = let(type_=View, domain=world.views)
    all_drawers = concatenate(views.drawers)  # <-- concatenate nested iterables into one iterable domain
    # Select the concatenated iterable only (single row expected)
    query1 = an(entity(all_drawers))

rows1 = list(query1.evaluate())
# rows1 is a list with a single UnificationDict; `all_drawers` maps to the concatenated list of drawers
assert len(rows1) == 1
assert {d.handle.name for d in rows1[0]} == {"Handle1", "Handle3"}

# Example 2: Test membership using in_/not_ with the concatenated iterable
with symbolic_mode():
    # A variable ranging over drawers in the world (simulate another source of drawers)
    d = Drawer(From([drawer1, drawer2]))
    views = CabinetLike(From(world.views))
    all_drawers = concatenate(views.drawers)
    # Find drawers that are NOT in the concatenated list (expect none in this tiny world)
    query2 = an(entity(d, not_(in_(d, all_drawers))))

rows2 = list(query2.evaluate())
assert len(rows2) == 0

with symbolic_mode():
    d = let(type_=Drawer, domain=[drawer1, drawer2])
    views = let(type_=View, domain=world.views)
    all_drawers = concatenate(views.drawers)
    # Find drawers that ARE in the concatenated list (expect both)
    query3 = an(entity(d, in_(d, all_drawers)))

rows3 = list(query3.evaluate())
assert {r.handle.name for r in rows3} == {"Handle1", "Handle3"}
