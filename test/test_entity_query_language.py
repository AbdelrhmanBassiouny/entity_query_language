from typing import Iterable

import pytest

from entity_query_language.entity import an, entity, set_of, let, the
from entity_query_language.failures import MultipleSolutionFound
from entity_query_language.symbolic import contains, in_, And, Or, Not, SymbolicMode
from .datasets import Handle, Body, Container, FixedConnection, PrismaticConnection, Drawer


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), body.name.startswith("Handle"))).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), contains(body.name, "Handle"))).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let(type_=Body, domain=world.bodies), in_("Handle", body.name))).evaluate()

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
                             contains(body.name, "Handle") & contains(body.name, '1'))).evaluate()

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
                             contains(body.name, "Handle1") | contains(body.name, 'Handle2'))).evaluate()

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
                             | contains(body.name, 'Container1'))).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."
    # assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                             Or(And(contains(body.name, "Handle"),
                                    contains(body.name, '1'))
                                , And(contains(body.name, 'Container'),
                                      contains(body.name, '1'))
                                )
                             )
                      ).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_generate_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                             Or(contains(body.name, "Handle"), contains(body.name, '1'))
                             , Or(contains(body.name, 'Container'), contains(body.name, '1'))
                             )
                      ).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_generate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_container1():
        yield from an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "n"), contains(body.name, '1')
                             , contains(body.name, 'C'))).evaluate()

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
                          container == fixed_connection.parent,
                          handle == fixed_connection.child,
                          container == prismatic_connection.child
                          )
                   ).evaluate()

    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert sol[container] == sol[fixed_connection].parent
        assert sol[handle] == sol[fixed_connection].child
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_the(handles_and_containers_world):
    world = handles_and_containers_world

    with pytest.raises(MultipleSolutionFound):
        handle = the(entity(body := let(type_=Handle, domain=world.bodies), body.name.startswith("Handle"))).evaluate()

    handle = the(entity(body := let(type_=Handle, domain=world.bodies), body.name.startswith("Handle1"))).evaluate()


def test_not_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle = an(entity(body := let(type_=Body, domain=world.bodies), Not(body.name.startswith("Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_comparator(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle = an(entity(body := let(type_=Body, domain=world.bodies), Not(contains(body.name, "Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_and(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1 = an(entity(body := let(type_=Body, domain=world.bodies),
                            Not(contains(body.name, "Handle") & contains(body.name, '1'))
                            )
                     ).evaluate()

    all_not_handle1 = list(not_handle1)
    assert len(all_not_handle1) == 5, "Should generate 5 bodies"
    assert all(h.name != "Handle1" for h in all_not_handle1), "All generated items should satisfy query"


def test_not_or(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1_or2 = an(entity(body := let(type_=Body, domain=world.bodies),
                                Not(contains(body.name, "Handle1") | contains(body.name, 'Handle2'))
                                )
                         ).evaluate()

    all_not_handle1_or2 = list(not_handle1_or2)
    assert len(all_not_handle1_or2) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Handle2"] for h in all_not_handle1_or2), "All generated items should satisfy query"


def test_not_and_or(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1_and_not_container1 = an(entity(body := let(type_=Body, domain=world.bodies),
                                               Not(Or(And(contains(body.name, "Handle"),
                                                          contains(body.name, '1'))
                                                      , And(contains(body.name, 'Container'),
                                                            contains(body.name, '1'))
                                                      ))
                                               )
                                        ).evaluate()

    all_not_handle1_and_not_container1 = list(not_handle1_and_not_container1)
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"] for h in
        all_not_handle1_and_not_container1), "All generated items should satisfy query"


def test_not_and_or_with_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1_and_not_container1 = an(entity(body := let(type_=Body, domain=world.bodies),
                                               Not(And(Or(body.name.startswith("Handle"),
                                                          body.name.endswith('1'))
                                                       , Or(body.name.startswith('Container'),
                                                            body.name.endswith('1'))
                                                       ))
                                               )
                                        ).evaluate()

    all_not_handle1_and_not_container1 = list(not_handle1_and_not_container1)
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"] for h in
        all_not_handle1_and_not_container1), "All generated items should satisfy query"


def test_generate_drawers(handles_and_containers_world):
    world = handles_and_containers_world

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    with SymbolicMode():
        solutions = an(entity(Drawer(handle=handle, container=container),
                              And(container == fixed_connection.parent,
                                  handle == fixed_connection.child,
                                  container == prismatic_connection.child))).evaluate()

    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"
