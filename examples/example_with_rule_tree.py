from entity_query_language import entity, an, let, and_, symbolic_mode, symbol, refinement, alternative, Add, rule_mode, \
    HasType, infer
from dataclasses import dataclass, field
from typing_extensions import List


# --- Domain model
@symbol
@dataclass
class Body:
    name: str
    size: int = 1


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
class RevoluteConnection(Connection):
    ...

@symbol
@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)

@symbol
@dataclass
class View:  # A common super-type for Drawer/Door/Wardrobe in this example
    ...


# Views we will construct symbolically
@dataclass
class Drawer(View):
    handle: Body
    container: Body


@dataclass
class Door(View):
    handle: Body
    body: Body


@dataclass
class Wardrobe(View):
    handle: Body
    body: Body
    container: Body


# --- Build a small "world"
container1, body2, body3, container2 = Container("Container1"), Body("Body2", size=2), Body("Body3"), Container("Container2")
handle1, handle2, handle3 = Handle("Handle1"), Handle("Handle2"), Handle("Handle3")
world = World(1, [container1, container2, body2, body3, handle1, handle2, handle3])

# Connections between bodies/handles
fixed_1 = FixedConnection(container1, handle1)
fixed_2 = FixedConnection(body2, handle2)
fixed_3 = FixedConnection(body3, handle3)
revolute_1 = RevoluteConnection(container2, body3)
world.connections = [fixed_1, fixed_2, fixed_3, revolute_1]

with symbolic_mode():
    # Declare the variables
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)
    views = let(type_=View)

    # Define aliases for convenience
    handle = fixed_connection.child
    body = fixed_connection.parent
    container = revolute_connection.parent

    # Describe base query
    # We use a single selected variable that we will Add to in the rule tree.
    query = infer(entity(views, HasType(fixed_connection.child, Handle)))

# --- Build the rule tree
with rule_mode(query):
    # Base conclusion: if a fixed connection exists between body and handle,
    # we consider it a Drawer by default.
    Add(views, Drawer(handle=handle, container=body))

    # Exception (refinement): If the body is "bigger" (size > 1), instead add a Door.
    # This refinement branch is more specific and can be seen as a refinement to the base rule.
    with refinement(body.size > 1):
        Add(views, Door(handle=handle, body=body))

        # Alternative refinement when the first refinement didn't fire: if the body is also connected to a parent
        # container via a revolute connection (alternative pattern), add a Wardrobe instead.
        with alternative(body == revolute_connection.child, container == revolute_connection.parent):
            Add(views, Wardrobe(handle=handle, body=body, container=container))

# query._render_tree_()

# Evaluate the rule tree
results = list(query.evaluate())

# The results include objects built from different branches of the rule tree.
# Depending on the world, you should observe a mix of Drawer, Door, and Wardrobe instances.
assert len(results) == 3
assert any(isinstance(v, Drawer) and v.handle.name == "Handle1" for v in results)
assert any(isinstance(v, Door) and v.handle.name == "Handle2" for v in results)
assert any(isinstance(v, Wardrobe) and v.handle.name == "Handle3" for v in results)