from dataclasses import dataclass
from typing_extensions import List
from entity_query_language import an, entity, let, symbolic_mode, symbol


@symbol
@dataclass(unsafe_hash=True)
class Body:
    name: str

@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]

world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2"), Body("Handle3")])

# Empty-conditions predicate form: just specify the type; all bodies are generated
with symbolic_mode():
    query_all = an(entity(body := Body(), domain=world.bodies))
assert len(list(query_all.evaluate())) == len(world.bodies)

# Predicate form with a specified property
with symbolic_mode():
    query_one = an(entity(Body(name="Handle1"), domain=world.bodies))
results_one = list(query_one.evaluate())
assert len(results_one) == 1 and results_one[0].name == "Handle1"