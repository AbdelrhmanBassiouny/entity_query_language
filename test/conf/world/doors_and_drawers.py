# ===================== Possible World Configurations ========================
from dataclasses import dataclass, field

from typing_extensions import List, Callable

from .base_config import WorldConf, BodyConf, Connection, FixedConnectionConf, PrismaticConnectionConf, \
    ContainerConf, RevoluteConnectionConf

from ...factories.world import create_world
from .handles_and_containers import Handle1, Handle2, Handle3, Container1


@dataclass
class Body1(ContainerConf):
    name: str = "Body1"


@dataclass
class Body2(ContainerConf):
    name: str = "Body2"
    size: int = 2


@dataclass
class Body3(ContainerConf):
    name: str = "Body3"



def bodies():
    return [
        Handle1(),
        Handle2(),
        Handle3(),
        Body1(),
        Body2(),
        Body3(),
        Container1()
    ]


@dataclass
class World(WorldConf):
    bodies: List[BodyConf] = field(default_factory=bodies, init=False)
    connections: List[Connection] = field(default_factory=lambda: [
        FixedConnectionConf(parent=Container1(), child=Handle1()),
        FixedConnectionConf(parent=Body2(), child=Handle2()),
        RevoluteConnectionConf(parent=Body3(), child=Handle3())
    ], init=False)
    factory_method: Callable = field(default=create_world, init=False)


