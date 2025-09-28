from dataclasses import dataclass, field
from typing import List

from entity_query_language import symbolic_mode, let, entity, an, From, the, for_all
from entity_query_language.predicate import symbol


# Minimal dataset for the example
@symbol
@dataclass
class Body:
    name: str


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
class Cabinet(View):
    container: Container


@symbol
@dataclass
class World:
    bodies: List[Body] = field(default_factory=list)
    views: List[View] = field(default_factory=list)


# Build a small world with two cabinets that share the same container
world = World()
container2 = Container(name="Container2")
world.bodies.extend([container2])

cab1 = Cabinet(container=container2)
cab2 = Cabinet(container=container2)
world.views = [cab1, cab2]

# Pick the container named "Container2"
with symbolic_mode():
    c = let(type_=Container, domain=world.bodies)
    the_cabinet_container = the(entity(c, c.name == "Container2"))

# Example 1: Universal constraint holds — all cabinets have that container
with symbolic_mode():
    cabinets = Cabinet(From(world.views))
    query1 = an(entity(
        the_cabinet_container,
        for_all(cabinets.container, the_cabinet_container == cabinets.container)
    ))

rows1 = list(query1.evaluate())
assert len(rows1) == 1
assert rows1[0].name == "Container2"

# Example 2: Universal constraint fails — require all cabinets to have a different container
with symbolic_mode():
    cabinets = Cabinet(From(world.views))
    query2 = an(entity(
        the_cabinet_container,
        for_all(cabinets.container, the_cabinet_container != cabinets.container)
    ))

rows2 = list(query2.evaluate())
assert len(rows2) == 0
