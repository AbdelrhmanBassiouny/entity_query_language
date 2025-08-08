from entity_query_language import entity, an, let, And, contains, the, MultipleSolutionFound, Or, Not
from dataclasses import dataclass
from typing_extensions import List


@dataclass(eq=False)
class Body:
    name: str

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(id(self))


@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]

    def __eq__(self, other):
        return self.id_ == other.id_

    def __hash__(self):
        return hash(self.id_)


world = World(1, [Body("Body1"), Body("Body2")])

results_generator = an(entity(body := let(type_=Body, domain=world.bodies),
                              And(contains(body.name, "2"), body.name.startswith("Body")))
                       )
results = list(results_generator)
assert len(results) == 1
assert results[0].name == "Body2"


world = World(1, [Body("Body1"), Body("Body2")])
body1 = the(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Body1")))
try:
    body = the(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Body")))
    assert False
except MultipleSolutionFound:
    pass

world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
result = an(entity(body := let(type_=Body, domain=world.bodies),
                   And(Or(body.name.startswith("C"), body.name.endswith("1")),
                       Or(body.name.startswith("H"), body.name.endswith("1"))
                       )
                   )
            )
results = list(result)
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"


result = an(entity(body := let(type_=Body, domain=world.bodies),
                   Not(And(Or(body.name.startswith("C"), body.name.endswith("1")),
                       Or(body.name.startswith("H"), body.name.endswith("1"))
                       )), show_tree=True
                   )
            )
results = list(result)
assert len(results) == 2
assert results[1].name == "Container2" and results[0].name == "Handle2"
