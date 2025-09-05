# Advanced Example: Rule Trees with Alternatives and Exceptions

This example demonstrates how to build a rule tree using refinement (specialization) and 
alternatives (mutually exclusive branches). It shows how to:
- Start from a base conclusion;
- Add a refined exception (more specific case) that overrides the base when a further condition is met;
- Add alternatives that apply under different conditions.

We will construct objects symbolically using symbolic_rule and Add, with let placeholders to describe relationships.

## Example Usage

```python
from entity_query_language import entity, an, let, and_, symbolic_mode, symbol, refinement, alternative, Add
from dataclasses import dataclass, field
from typing_extensions import List


# --- Domain model
@dataclass
class Body:
    name: str
    size: int = 1


@dataclass
class Connection:
    parent: Body
    child: Body


@dataclass
class Fixed(Connection):
    ...


@dataclass
class Revolute(Connection):
    ...


@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


@dataclass
class View:  # A common super-type for Drawer/Door/Wardrobe in this example
    ...


# Views we will construct symbolically
@symbol
@dataclass
class Drawer(View):
    handle: Body
    container: Body


@symbol
@dataclass
class Door(View):
    handle: Body
    body: Body


@symbol
@dataclass
class Wardrobe(View):
    handle: Body
    body: Body
    container: Body


# --- Build a small "world"
container1, body2, body3, container2 = Body("Container1"), Body("Body2", size=2), Body("Body3"), Body("Container2")
handle1, handle2, handle3 = Body("Handle1"), Body("Handle2"), Body("Handle3")
world = World(1, [container1, container2, body2, body3, handle1, handle2, handle3])

# Connections between bodies/handles
fixed_1 = Fixed(container1, handle1)
fixed_2 = Fixed(body2, handle2)
fixed_3 = Fixed(body3, handle3)
revolute_1 = Revolute(container2, body3)
world.connections = [fixed_1, fixed_2, fixed_3, revolute_1]

# --- Placeholders
world = let("world", type_=World, domain=world)
body = let("body", type_=Body, domain=world.bodies)
container = let("container", type_=Body, domain=world.bodies)
handle = let("handle", type_=Body, domain=world.bodies)
fixed_connection = let("fixed_connection", type_=Fixed, domain=world.connections)
revolute_connection = let("revolute_connection", type_=Revolute, domain=world.connections)

views = let("views", type_=View)
# --- Describe base query
# We use a single selected variable that we will Add to in the rule tree.
query = an(entity(views,
                  body == fixed_connection.parent,
                  handle == fixed_connection.child))

# --- Build the rule tree
with symbolic_mode(query):
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

print(f"Results: {results}")

# The results include objects built from different branches of the rule tree.
# Depending on the world, you should observe a mix of Drawer, Door, and Wardrobe instances.
assert len(results) == 3
assert any(isinstance(v, Drawer) and v.handle.name == "Handle1" for v in results)
assert any(isinstance(v, Door) and v.handle.name == "Handle2" for v in results)
assert any(isinstance(v, Wardrobe) and v.handle.name == "Handle3" for v in results)
```

### Notes
- refinement(*conditions): narrows the context with an additional condition (like an exception/specialization).
- alternative(*conditions): introduces a sibling branch with its own conditions; only if those are satisfied will that branch contribute conclusions.
- Add(target, value): materializes a conclusion into the selected variable (here, a collection-like placeholder views).
- Remember to always pass a mandatory name to let(name, type_, domain=...).
