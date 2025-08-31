from entity_query_language import entity, an, let, and_, contains, the, MultipleSolutionFound, or_, not_, in_, set_of, \
    symbolic_mode, symbol
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

results_generator = an(entity(body := let("body", type_=Body, domain=world.bodies),
                              and_(contains(body.name, "2"), body.name.startswith("Body"))
                              )).evaluate()
results = list(results_generator)
assert len(results) == 1
assert results[0].name == "Body2"


world = World(1, [Body("Body1"), Body("Body2")])
body1 = the(entity(body := let("body", type_=Body, domain=world.bodies), body.name.startswith("Body1"))).evaluate()
try:
    body = the(entity(body := let("body", type_=Body, domain=world.bodies), body.name.startswith("Body"))).evaluate()
    assert False
except MultipleSolutionFound:
    pass

world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
result = an(entity(body := let("body", type_=Body, domain=world.bodies),
                   and_(or_(body.name.startswith("C"), body.name.endswith("1")),
                        or_(body.name.startswith("H"), body.name.endswith("1"))
                        )
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"


result = an(entity(body := let("body", type_=Body, domain=world.bodies),
                   not_(and_(or_(body.name.startswith("C"), body.name.endswith("1")),
                            or_(body.name.startswith("H"), body.name.endswith("1"))
                            ))
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[0].name == "Container2" and results[1].name == "Handle2"

world2 = World(2, [Body("Container3"), Body("Handle3"), Body("Handle2")])

result = an(entity(body := let("body", type_=Body, domain=world2.bodies),
                   and_(body.name.startswith('H'), in_(body, world.bodies))
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

parent_container = let("parent_container", type_=Body, domain=world.bodies)
prismatic_connection = let("prismatic_connection", type_=Prismatic, domain=world.connections)
drawer_body = let("drawer_body", type_=Body, domain=world.bodies)
fixed_connection = let("fixed_connection", type_=Fixed, domain=world.connections)
handle = let("handle", type_=Body, domain=world.bodies)

drawer_kinematic_tree = (parent_container, prismatic_connection, drawer_body, fixed_connection, handle)
result = an(set_of(drawer_kinematic_tree,
                   and_(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                        drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                   )
            ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0][parent_container].name == "Container1"
assert results[0][drawer_body].name == "Container2"
assert results[0][handle].name == "Handle2"


# Write the query body
with symbolic_mode():
    result = an(entity(Drawer(handle=handle, body=drawer_body),
                       and_(parent_container == prismatic_connection.parent, drawer_body == prismatic_connection.child,
                            drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                       )
                ).evaluate()
results = list(result)
assert len(results) == 1
assert results[0].body.name == "Container2"
assert results[0].handle.name == "Handle2"
