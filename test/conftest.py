import pytest

from entity_query_language.symbolic import Variable
from .conf.world.handles_and_containers import World as HandlesAndContainersWorld
from .conf.world.doors_and_drawers import World as DoorsAndDrawersWorld
from .datasets import *


@pytest.fixture
def handles_and_containers_world() -> World:
    return HandlesAndContainersWorld().create()

@pytest.fixture
def doors_and_drawers_world() -> World:
    return DoorsAndDrawersWorld().create()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    # Setup: runs before each test
    yield
    # Teardown: runs after each test
    for c in Variable._cache_.values():
        c.clear()
    Variable._cache_.clear()

