# Example with rule inference

In this example, we show the how EQL allows for straight forward inference (i.e. rule-based reasoning) for 
classification of relational concepts.

In the previous example, we wrote an advanced query that joined multiple sources together to find the kinematic tree of
a drawer. Now, we will show how to easily construct the Drawer instance(s) from the found kinematic tree(s).

Here we introduce the `@symbol` decorator that allows us to create symbolic instances of a class without invoking its
constructor. This is only possible in the `SymbolicMode` to avoid side effects.

## Example Usage

```python
from entity_query_language import entity, let, an, and_, in_, set_of, symbolic_mode, symbol
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
class Prismatic(Connection):
    ...


@dataclass
class Fixed(Connection):
    ...


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
world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
c1_c2 = Prismatic(world.bodies[0], world.bodies[1])
c2_h2 = Fixed(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

# Query for the kinematic tree of the drawer which has more than one component.
# Declare the placeholders
parent_container = let("parent_container", type_=Body, domain=world.bodies)
prismatic_connection = let("prismatic_connection", type_=Prismatic, domain=world.connections)
drawer_body = let("drawer_body", type_=Body, domain=world.bodies)
fixed_connection = let("fixed_connection", type_=Fixed, domain=world.connections)
handle = let("handle", type_=Body, domain=world.bodies)

# Write the query body
with symbolic_mode():
    result = an(entity(Drawer(handle=handle, body=drawer_body),
                       and_(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                            drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                       )
                ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0].body.name == "Container2"
assert results[0].handle.name == "Handle2"
```

The key difference between this example and the previous one is that our entity is now a `Drawer` instance that 
gets constructed from the components of the kinematic tree that is found by the query conditions.

To do that, we had to be in the `SymbolicMode` to allow for symbolic creation of the `Drawer` instance without
invoking the `Drawer` constructor. This is done by overriding the `__new__` method of the `Drawer` class with the
`@symbol` decorator.
