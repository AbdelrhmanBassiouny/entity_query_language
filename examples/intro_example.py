from dataclasses import dataclass

from typing_extensions import List

from entity_query_language import entity, an, contains, symbolic_mode, symbol, From


@symbol
@dataclass
class Body:
    name: str


@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

with symbolic_mode():
    body = Body(From(world.bodies))
    query = an(entity(body, contains(body.name, "2"),
                      body.name.startswith("Body"))
               )
results = list(query.evaluate())
assert len(results) == 1
assert results[0].name == "Body2"