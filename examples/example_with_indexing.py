from dataclasses import dataclass
from typing_extensions import Dict, List
from entity_query_language import an, entity, let, symbolic_mode, symbol


@symbol
@dataclass
class Body:
    name: str
    props: Dict[str, int]


@dataclass
class World:
    bodies: List[Body]


world = World([
    Body("Body1", {"score": 1}),
    Body("Body2", {"score": 2}),
])

with symbolic_mode():
    b = let(type_=Body, domain=world.bodies)
    # use indexing on a dict field
    query = an(entity(b, b.props["score"] == 2))

results = list(query.evaluate())
assert len(results) == 1 and results[0].name == "Body2"
