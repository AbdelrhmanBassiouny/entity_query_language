from ..datasets import World, Handle, Container, FixedConnection, PrismaticConnection, RevoluteConnection, Body
from ..conf.world.base_config import WorldConf, HandleConf, ContainerConf, FixedConnectionConf, \
    PrismaticConnectionConf, RevoluteConnectionConf


def create_world(world_conf: WorldConf) -> World:
    world = World()

    for body in world_conf.bodies:
        if isinstance(body, HandleConf):
            world.bodies.append(Handle(body.name, size=body.size, world=world))
        elif isinstance(body, ContainerConf):
            world.bodies.append(Container(body.name, size=body.size, world=world))
        else:
            world.bodies.append(Body(body.name, size=body.size, world=world))
    for connection in world_conf.connections:
        parent = next((b for b in world.bodies if b.name == connection.parent.name), None)
        child = next((b for b in world.bodies if b.name == connection.child.name), None)
        if parent and child:
            if isinstance(connection, FixedConnectionConf):
                connection_cls = FixedConnection
            elif isinstance(connection, PrismaticConnectionConf):
                connection_cls = PrismaticConnection
            elif isinstance(connection, RevoluteConnectionConf):
                connection_cls = RevoluteConnection
            else:
                raise ValueError(f"Unknown connection type: {connection}")
            world.connections.append(connection_cls(parent=parent, child=child, world=world))

    return world
