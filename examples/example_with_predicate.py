from dataclasses import dataclass
from typing_extensions import List

from entity_query_language import entity, let, an, predicate, symbolic_mode, symbol, Predicate


@symbol
@dataclass
class Body:
    name: str


@dataclass
class Handle(Body):
    pass


@dataclass
class Container(Body):
    pass

@symbol
@dataclass
class World:
    id_: int
    bodies: List[Body]


# Define a reusable predicate: returns True if a body is a handle by name convention
@predicate
def is_handle(body_: Body) -> bool:
    return body_.name.startswith("Handle")

@dataclass
class HasThreeInItsName(Predicate):
    body: Body

    def __call__(self):
        return '3' in self.body.name


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
            body := let(type_=Body, domain=world.bodies),
            is_handle(body_=body),  # use the predicate just like any other condition
            HasThreeInItsName(body)
        )
    )

# Evaluate and inspect the results
results = list(query.evaluate())
assert len(results) == 1
assert isinstance(results[0], Handle)
assert results[0].name == "Handle3"