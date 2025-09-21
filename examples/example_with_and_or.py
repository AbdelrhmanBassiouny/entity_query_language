from dataclasses import dataclass

from typing_extensions import List

from entity_query_language import entity, an, or_, symbolic_mode, symbol, let


@symbol
@dataclass(unsafe_hash=True)
class Body:
    name: str


@symbol
@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]


# Construct a world with some bodies
world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])

with symbolic_mode():
    # Declare the variables
    body = let(type_=Body, domain=world.bodies)
    # Construct the query
    query = an(entity(body,
                      or_(body.name.startswith("C"), body.name.endswith("1")),
                      or_(body.name.startswith("H"), body.name.endswith("1"))
                      )
               )
results = list(query.evaluate())
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"
