from dataclasses import dataclass
from typing_extensions import List
from entity_query_language import an, entity, let, symbolic_mode, symbol


@symbol
@dataclass(unsafe_hash=True)
class Body:
    name: str

@symbol
@dataclass
class Connection:
    parent: Body
    child: Body

@symbol
@dataclass
class FixedConnection(Connection):
    ...

@symbol
@dataclass
class PrismaticConnection(Connection):
    ...

@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection]

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
class Drawer:
    handle: Handle
    container: Container

# Build a small world with two drawer configurations
handle1 = Handle("Handle1"); handle3 = Handle("Handle3")
container1 = Container("Container1"); container3 = Container("Container3")
fixed1 = FixedConnection(parent=container1, child=handle1)
prism1 = PrismaticConnection(parent=container1, child=container1)  # not used directly but keeps structure
fixed3 = FixedConnection(parent=container3, child=handle3)
prism3 = PrismaticConnection(parent=container3, child=container3)
world = World(1, [container1, container3, handle1, handle3], [fixed1, prism1, fixed3, prism3])

# Pure predicate-form rule: construct Drawer by matching sub-graphs
with symbolic_mode():
    query = an(
        entity(
            Drawer(handle=an(entity(handle := Handle(), domain=world.bodies)),
                   container=an(entity(container := Container(), domain=world.bodies))),
            an(entity(FixedConnection(parent=container, child=handle), domain=world.connections)),
            an(entity(PrismaticConnection(child=container), domain=world.connections))
        )
    )

solutions = list(query.evaluate())
assert len(solutions) == 2
assert { (d.handle.name, d.container.name) for d in solutions } == { ("Handle1", "Container1"), ("Handle3", "Container3") }