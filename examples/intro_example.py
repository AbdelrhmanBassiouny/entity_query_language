from entity_query_language import entity, an, let
from dataclasses import dataclass
from typing_extensions import List


@dataclass(unsafe_hash=True)
class Body:
    name: str


@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

results_generator = an(entity(body := let(type_=Body, domain=world.bodies), body.name == "Body2"))
results = list(results_generator)
assert results[0].name == "Body2"
