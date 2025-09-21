# Example with rule inference

In this example, we show the how EQL allows for straight forward inference (i.e. rule-based reasoning) for 
classification of relational concepts.

In the previous example, we wrote an advanced query that joined multiple sources together to find the kinematic tree of
a drawer. Now, we will show how to easily construct the Drawer instance(s) from the found kinematic tree(s).

Here we introduce the `infer` function that allows for inferring new entities from existing ones based on a set of
conditions. In addition, we show how to use the `rule_mode` to allow for rule-based reasoning.

## Example Usage

```python
from dataclasses import dataclass, field

from typing_extensions import List

from entity_query_language import entity, symbol, infer, From, HasType, rule_mode, let


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
class PrismaticConnection(Connection):
    ...


@dataclass
class FixedConnection(Connection):
    ...


@symbol
@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


@symbol
@dataclass
class Drawer:
    handle: Body
    body: Body


# Create the world with its bodies and connections
world = World(1, [Container("Container1"), Container("Container2"), Handle("Handle1"), Handle("Handle2")])
c1_c2 = PrismaticConnection(world.bodies[0], world.bodies[1])
c2_h2 = FixedConnection(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

# A rule for finding drawers.
with rule_mode():
    # Declare the variables
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    parent_container = prismatic_connection.parent
    drawer_body = fixed_connection.parent
    handle = fixed_connection.child

    # Write the rule body
    rule = infer(entity(Drawer(handle=handle, body=drawer_body),
                        HasType(prismatic_connection.parent, Container),
                        HasType(fixed_connection.child, Handle),
                        prismatic_connection.child == fixed_connection.parent))

results = list(rule.evaluate())
assert len(results) == 1
assert results[0].body.name == "Container2"
assert results[0].handle.name == "Handle2"

```

The key difference between this example and the previous one is that our entity is now a `Drawer` instance that 
gets inferred from the components of the kinematic tree that is found by the query conditions.

To do that, we had to be in the `rule_mode` to allow for symbolic inference of the `Drawer` instance without
invoking the `Drawer` constructor and without implying that we are querying for drawers.
This is done by overriding the `__new__` method of the `Drawer` class with the `@symbol` decorator.
