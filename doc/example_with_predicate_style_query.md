# Example: Predicate (Functional) Style Queries and Rules

This example shows how to write queries and rules using the predicate form.

The predicate form lets you construct entities directly with their fields as predicate constraints,
within `symbolic_mode()` instead of first defining a `let` variable and then adding constraints to it in
the `entity(...)` term, this would appeal more to users who prefer functional programming style.

## Query in predicate form

Here we show simple queries using the predicate form to find `Body` objects in a `World`.

```python
from dataclasses import dataclass
from typing_extensions import List
from entity_query_language import an, entity, let, symbolic_mode, symbol, From

@symbol
@dataclass(unsafe_hash=True)
class Body:
    name: str

@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]

world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2"), Body("Handle3")])

# Empty-conditions predicate form: just specify the type; all bodies are generated
with symbolic_mode():
    query_all_bodies = Body(From(world.bodies))
assert len(list(query_all_bodies.evaluate())) == len(world.bodies)

# Predicate form with a specified property
with symbolic_mode():
    query_one = Body(From(world.bodies), name="Handle1")
results_one = list(query_one.evaluate())
assert len(results_one) == 1 and results_one[0].name == "Handle1"
```

## Drawer rule in predicate form

Here we construct `Drawer` objects from matching kinematic sub-graphs using only predicate-style entity terms.

```python
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
```

Notes:
- The key pattern is to place class constructors with desired field constraints inside `entity(...)` while inside `symbolic_mode()`.
- Ensure that the constructed entities have there classes decorated with `@symbol` to enable symbolic construction.
- Use nested `an(entity(..., domain=...))` terms to bind intermediate symbolic objects and reuse them across conditions.
