from entity_query_language import entity, an, let, And, contains, the, MultipleSolutionFound, Or, Not, in_, set_of, \
    SymbolicRule, symbol
from dataclasses import dataclass, field
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
                       ).evaluate()
results = list(results_generator)
assert len(results) == 1
assert results[0].name == "Body2"


world = World(1, [Body("Body1"), Body("Body2")])
body1 = the(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Body1"))).evaluate()
try:
    body = the(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Body"))).evaluate()
    assert False
except MultipleSolutionFound:
    pass

world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
result = an(entity(body := let(type_=Body, domain=world.bodies),
                   And(Or(body.name.startswith("C"), body.name.endswith("1")),
                       Or(body.name.startswith("H"), body.name.endswith("1"))
                       )
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"


result = an(entity(body := let(type_=Body, domain=world.bodies),
                   Not(And(Or(body.name.startswith("C"), body.name.endswith("1")),
                       Or(body.name.startswith("H"), body.name.endswith("1"))
                       ))
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[1].name == "Container2" and results[0].name == "Handle2"

world2 = World(2, [Body("Container3"), Body("Handle3"), Body("Handle2")])

result = an(entity(body := let(type_=Body, domain=world2.bodies),
                   And(body.name.startswith('H'), in_(body, world.bodies))
                   )
            ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0].name == "Handle2"


@dataclass
class Connection:
    parent: Body
    child: Body


@dataclass
class Prismatic(Connection):
    ...


@dataclass
class Fixed(Connection):
    ...


@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


@symbol
@dataclass
class Drawer:
    handle: Body
    body: Body



world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
c1_c2 = Prismatic(world.bodies[0], world.bodies[1])
c2_h2 = Fixed(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

parent_container = let(type_=Body, domain=world.bodies)
prismatic_connection = let(type_=Prismatic, domain=world.connections)
drawer_body = let(type_=Body, domain=world.bodies)
fixed_connection = let(type_=Fixed, domain=world.connections)
handle = let(type_=Body, domain=world.bodies)

drawer_kinematic_tree = (parent_container, prismatic_connection, drawer_body, fixed_connection, handle)
result = an(set_of(drawer_kinematic_tree,
                   And(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                       drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                   )
            ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0][parent_container].name == "Container1"
assert results[0][drawer_body].name == "Container2"
assert results[0][handle].name == "Handle2"


# Write the query body
with SymbolicRule():
    result = an(entity(Drawer(handle=handle, body=drawer_body),
                       And(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                           drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                       )
                ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0].body.name == "Container2"
assert results[0].handle.name == "Handle2"
