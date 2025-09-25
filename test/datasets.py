from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from typing_extensions import List, Optional, ClassVar

from entity_query_language import symbol


@dataclass(unsafe_hash=True)
class WorldEntity:
    world: Optional[World] = field(default=None, kw_only=True, repr=False, hash=False)


@symbol
@dataclass(unsafe_hash=True)
class Body(WorldEntity):
    name: str
    size: int = field(default=1)


@symbol
@dataclass(unsafe_hash=True)
class Handle(Body):
    ...


@symbol
@dataclass(unsafe_hash=True)
class Container(Body):
    ...


@symbol
@dataclass(unsafe_hash=True)
class Connection(WorldEntity):
    parent: Body
    child: Body


@symbol
@dataclass(unsafe_hash=True)
class FixedConnection(Connection):
    ...


@symbol
@dataclass(unsafe_hash=True)
class PrismaticConnection(Connection):
    ...


@symbol
@dataclass(unsafe_hash=True)
class RevoluteConnection(Connection):
    ...


@symbol
@dataclass
class World:
    id: int = field(default=0)
    bodies: List[Body] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    views: List[View] = field(default_factory=list, repr=False)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, World):
            return False
        return self.id == other.id

@symbol
@dataclass(unsafe_hash=True)
class View(WorldEntity):
    ...


@symbol
@dataclass
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None

    def __hash__(self):
        return hash((self.__class__.__name__, self.handle, self.container))

    def __eq__(self, other):
        if not isinstance(other, Drawer):
            return False
        return self.handle == other.handle and self.container == other.container and self.world == other.world


@symbol
@dataclass
class Cabinet(View):
    container: Container
    drawers: List[Drawer] = field(default_factory=list)

    def __hash__(self):
        return hash((self.__class__.__name__, self.container))

    def __eq__(self, other):
        if not isinstance(other, Cabinet):
            return False
        return self.container == other.container and self.drawers == other.drawers and self.world == other.world


@symbol
@dataclass(unsafe_hash=True)
class Door(View):
    handle: Handle
    body: Body


@symbol
@dataclass(unsafe_hash=True)
class Wardrobe(View):
    handle: Handle
    body: Body
    container: Container
