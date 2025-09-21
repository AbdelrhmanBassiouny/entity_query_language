from dataclasses import dataclass

from typing_extensions import List

from entity_query_language import entity, let, the, MultipleSolutionFound, symbol, symbolic_mode


@symbol
@dataclass
class Body:
    name: str

@symbol
@dataclass
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

with symbolic_mode():
    body1 = the(entity(body := let(type_=Body, domain=world.bodies),
                       body.name.startswith("Body1"))).evaluate()
    try:
        body = the(entity(body := let(type_=Body, domain=world.bodies),
                          body.name.startswith("Body"))).evaluate()
        assert False
    except MultipleSolutionFound:
        pass
