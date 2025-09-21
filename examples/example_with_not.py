from entity_query_language import entity, let, an, and_, or_, not_, symbol, symbolic_mode
from dataclasses import dataclass
from typing_extensions import List


@symbol
@dataclass
class Body:
    name: str


@symbol
@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])

with symbolic_mode():
    query = an(entity(body := let(type_=Body, domain=world.bodies),
                       not_(and_(or_(body.name.startswith("C"), body.name.endswith("1")),
                                 or_(body.name.startswith("H"), body.name.endswith("1"))
                                 ))
                       )
                )
results = list(query.evaluate())
assert len(results) == 2
assert results[0].name == "Container2" and results[1].name == "Handle2"