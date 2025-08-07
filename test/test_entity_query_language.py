import pytest

from entity_query_language.entity import an, entity, set_of, let, the
from entity_query_language.failures import MultipleSolutionFound
from entity_query_language.symbolic import contains, in_
from .datasets import Handle, Body, Container, FixedConnection, PrismaticConnection, Drawer


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Handle")))

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), contains(body.name, "Handle")))

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), in_("Handle", body.name)))

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                                 contains(body.name, "Handle") & contains(body.name, '1')))

    handles = list(generate_handles())
    assert len(handles) == 1, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                                 contains(body.name, "Handle1") | contains(body.name, 'Handle2')))

    handles = list(generate_handles())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_multi_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles_and_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                                 contains(body.name, "Handle1")
                                 | contains(body.name, 'Handle2')
                                 | contains(body.name, 'Container1')))

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."
    # assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                                 (contains(body.name, "Handle") & contains(body.name, '1'))
                                 | (contains(body.name, 'Container') & contains(body.name, '1'))))

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_generate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "n") & contains(body.name, '1')
                             & contains(body.name, 'C')))

    all_solutions = list(generate_container1())
    assert len(all_solutions) == 1, "Should generate one container."
    assert isinstance(all_solutions[0], Container), "The generated item should be of type Container."
    assert all_solutions[0].name == "Container1"


def test_generate_with_more_than_one_source(handles_and_containers_world):
    world = handles_and_containers_world

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)

    solutions = an(set_of(drawer_components,
                       (container == fixed_connection.parent) & (handle == fixed_connection.child) &
                       (container == prismatic_connection.child)))

    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert sol[container] == sol[fixed_connection].parent
        assert sol[handle] == sol[fixed_connection].child
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_the(handles_and_containers_world):
    world = handles_and_containers_world

    with pytest.raises(MultipleSolutionFound):
        handle = the(entity(body := let(type_=Handle, domain=world.bodies), body.name.startswith("Handle")))

    handle = the(entity(body := let(type_=Handle, domain=world.bodies), body.name.startswith("Handle1")))

