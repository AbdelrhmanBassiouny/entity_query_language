import pytest

from .conf.world.handles_and_containers import World as HandlesAndContainersWorld
from .conf.world.doors_and_drawers import World as DoorsAndDrawersWorld
from .datasets import *


@pytest.fixture
def handles_and_containers_world() -> World:
    return HandlesAndContainersWorld().create()

@pytest.fixture
def doors_and_drawers_world() -> World:
    return DoorsAndDrawersWorld().create()

