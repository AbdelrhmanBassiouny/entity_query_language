# Example with `Joining` Multiple Sources.

In this example, we show the how EQL can perform complex queries that require joining of multiple sources 
(equivalent to tables in a structured database) without ever mentioning join or how to join, instead it is implicit
in the conditions of the query.

This allows for a minimal query description that only contains the high level logic.

## Example Usage

```python
from entity_query_language import entity, let, an, and_, in_, set_of
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
drawer_kinematic_tree = (parent_container, prismatic_connection, drawer_body, fixed_connection, handle)

# Write the query body
result = an(set_of(drawer_kinematic_tree,
                   and_(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                        drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                   )
            ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0][parent_container].name == "Container1"
assert results[0][drawer_body].name == "Container2"
assert results[0][handle].name == "Handle2"
```

In the above example we want to find all drawers and their components by describing their kinematic tree using a
conjunction (AND operation) of conditions that show how the components are connected to each other to form the kinematic
tree.
