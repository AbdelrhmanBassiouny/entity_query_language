from dataclasses import dataclass
from typing_extensions import List

from entity_query_language import entity, let, an, predicate, symbolic_mode


@dataclass(unsafe_hash=True)
class Body:
    name: str


@dataclass(unsafe_hash=True)
class Handle(Body):
    pass


@dataclass(unsafe_hash=True)
class Container(Body):
    pass


@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]


# Define a reusable predicate: returns True if a body is a handle by name convention
@predicate
def is_handle(body_: Body) -> bool:
    return body_.name.startswith("Handle")


# Sample world containing containers and handles
world = World(
    1,
    [
        Container("Container1"),
        Container("Container2"),
        Handle("Handle1"),
        Handle("Handle2"),
        Handle("Handle3"),
    ],
)

# Build the query using the predicate inside symbolic mode
with symbolic_mode():
    query = an(
        entity(
            body := let("body", type_=Body, domain=world.bodies),
            is_handle(body_=body),  # use the predicate just like any other condition
        )
    )

# Evaluate and inspect the results
results = list(query.evaluate())
assert len(results) == 3
assert all(isinstance(h, Handle) for h in results)