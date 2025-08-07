import pytest

from .conf.world.handles_and_containers import HandlesAndContainersWorld
from .datasets import *


@pytest.fixture
def handles_and_containers_world() -> World:
    return HandlesAndContainersWorld().create()

