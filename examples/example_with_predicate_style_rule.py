from dataclasses import dataclass

from typing_extensions import List

from entity_query_language import an, entity, symbolic_mode, symbol, From, a, rule_mode, infer, HasType


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
handle1 = Handle("Handle1");
handle3 = Handle("Handle3")
container1 = Container("Container1");
container3 = Container("Container3")
fixed1 = FixedConnection(parent=container1, child=handle1)
prism1 = PrismaticConnection(parent=container1, child=container1)  # not used directly but keeps structure
fixed3 = FixedConnection(parent=container3, child=handle3)
prism3 = PrismaticConnection(parent=container3, child=container3)
world = World(1, [container1, container3, handle1, handle3], [fixed1, prism1, fixed3, prism3])

# Pure predicate-form rule: construct Drawer by matching sub-trees
with rule_mode():
    # Declare the variables
    prismatic_connection = PrismaticConnection(From(world.connections))
    fixed_connection = FixedConnection(From(world.connections))

    # Define some aliases for convenience
    parent_container = prismatic_connection.parent
    drawer_body = fixed_connection.parent
    handle = fixed_connection.child

    # Write the rule body
    rule = infer(entity(Drawer(handle=handle, container=drawer_body),
                        HasType(parent_container, Container),
                        HasType(handle, Handle),
                        drawer_body == prismatic_connection.child))

solutions = list(rule.evaluate())
assert len(solutions) == 2
assert {(d.handle.name, d.container.name) for d in solutions} == {("Handle1", "Container1"), ("Handle3", "Container3")}
