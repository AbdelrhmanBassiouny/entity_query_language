from dataclasses import dataclass, field

from typing_extensions import List

from entity_query_language import entity, an, let, set_of, symbol, symbolic_mode, a, HasType


@symbol
@dataclass
class Body:
    name: str


@dataclass
class Container(Body):
    ...


@dataclass
class Handle(Body):
    ...


@symbol
@dataclass
class Connection:
    parent: Body
    child: Body


@dataclass
class FixedConnection(Connection):
    ...


@dataclass
class PrismaticConnection(Connection):
    ...


@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


# Sample data
bodies = [Container("Container1"), Handle("Handle1"), Container("Container2"), Handle("Handle2"),
          Container("Container3")]
fixed_1 = FixedConnection(parent=bodies[0], child=bodies[1])  # Container1 -> Handle1
prismatic_1 = PrismaticConnection(parent=bodies[4], child=bodies[0])  # Container2 -> Container1
fixed_2 = FixedConnection(parent=bodies[2], child=bodies[3])  # Container2 -> Handle2
prismatic_2 = PrismaticConnection(parent=bodies[4], child=bodies[2])  # Container1 -> Container2
world = World(1, bodies=bodies, connections=[fixed_1, prismatic_1, fixed_2, prismatic_2])

with symbolic_mode():
    # Construct the necessary variables
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)

    # Declare selected variables that will be outputted by the query
    container = fixed_connection.parent
    handle = fixed_connection.child
    drawer_components = (prismatic_connection, container, fixed_connection, handle)

    # Nested Query construction
    containers_that_have_handles = a(set_of((container, handle),
                                            HasType(fixed_connection.parent, Container),
                                            HasType(fixed_connection.child, Handle)
                                            )
                                     )
    containers_that_can_translate = an(entity(container, container == prismatic_connection.child))

    nested_query = a(set_of(drawer_components, containers_that_have_handles & containers_that_can_translate))

nested_query_results = list(nested_query.evaluate())
assert len(nested_query_results) == 2, "Should generate components for 2 drawers"
