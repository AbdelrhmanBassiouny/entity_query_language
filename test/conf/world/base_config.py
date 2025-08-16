from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import List, TYPE_CHECKING, Callable, Any


@dataclass
class BodyConf:
    name: str = MISSING
    size: int = field(default=1)


@dataclass
class HandleConf(BodyConf):
    ...


@dataclass
class ContainerConf(BodyConf):
    ...


@dataclass
class Connection:
    parent: BodyConf = MISSING
    child: BodyConf = MISSING


@dataclass
class FixedConnectionConf(Connection):
    pass


@dataclass
class PrismaticConnectionConf(Connection):
    pass


@dataclass
class RevoluteConnectionConf(Connection):
    pass


@dataclass
class CaseConf:
    factory_method: Callable[[Any], Any] = MISSING

    def create(self) -> Any:
        return self.factory_method()


@dataclass
class WorldConf(CaseConf):
    bodies: List[BodyConf] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
