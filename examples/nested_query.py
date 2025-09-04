from entity_query_language import entity, an, let, set_of
from dataclasses import dataclass, field
from typing_extensions import List


@dataclass
class Body:
    name: str


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
bodies = [Body("Container1"), Body("Handle1"), Body("Container2"), Body("Handle2"), Body("Container3")]
fixed_1 = FixedConnection(parent=bodies[0], child=bodies[1])  # Container1 -> Handle1
prismatic_1 = PrismaticConnection(parent=bodies[4], child=bodies[0])  # Container2 -> Container1
fixed_2 = FixedConnection(parent=bodies[2], child=bodies[3])  # Container2 -> Handle2
prismatic_2 = PrismaticConnection(parent=bodies[4], child=bodies[2])  # Container1 -> Container2
world = World(1, bodies=bodies, connections=[fixed_1, prismatic_1, fixed_2, prismatic_2])

# Variables
container = let("container", type_=Body, domain=world.bodies)
handle = let("handle", type_=Body, domain=world.bodies)
fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
drawer_components = (container, handle)

# Original (flat) query
original_query = an(set_of(drawer_components,
                           container == fixed_connection.parent,
                           handle == fixed_connection.child,
                           container == prismatic_connection.child
                           )
                    )

original_query_results = list(original_query.evaluate())
assert len(original_query_results) == 2, "Should generate 2 drawer components"

original_results = list(original_query.evaluate())

# Nested version
containers_that_have_handles = an(set_of((container, handle),
                                         container == fixed_connection.parent,
                                         handle == fixed_connection.child
                                         )
                                  )
containers_that_can_translate = an(entity(container, container == prismatic_connection.child))
nested_query = an(set_of(drawer_components, containers_that_have_handles & containers_that_can_translate))

nested_query_results = list(nested_query.evaluate())
assert len(nested_query_results) == 2, "Should generate 2 drawer components"
assert nested_query_results == original_query_results, "Should generate same results"