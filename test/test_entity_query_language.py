import pytest

from entity_query_language import And, Or, Not, contains, in_
from entity_query_language.entity import an, entity, set_of, let, the
from entity_query_language.failures import MultipleSolutionFound
from entity_query_language.symbolic import SymbolicRule, Add, refinement, alternative, _cache_search_count, \
    _cache_match_count, _cache_enter_count
from entity_query_language.utils import render_tree
from .datasets import Handle, Body, Container, FixedConnection, PrismaticConnection, Drawer, RevoluteConnection, Door, \
    View, World, Wardrobe


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(
            entity(body := let("body", type_=Body, domain=world.bodies), body.name.startswith("Handle"))).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies),
                             contains(body.name, "Handle"))).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies), in_("Handle", body.name))).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies),
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
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies),
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
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies),
                             contains(body.name, "Handle1")
                             | contains(body.name, 'Handle2')
                             | contains(body.name, 'Container1'))).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."


def test_generate_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        yield from an(entity(body := let("body", type_=Body, domain=world.bodies),
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
        query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                             Or(contains(body.name, "Handle"), contains(body.name, '1'))
                             , Or(contains(body.name, 'Container'), contains(body.name, '1'))
                             )
                      )
        # query._render_tree_()
        yield from query.evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_generate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_container1():
        query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                             contains(body.name, "n"), contains(body.name, '1')
                             , contains(body.name, 'C')))

        # query._render_tree_()
        yield from query.evaluate()
    all_solutions = list(generate_container1())
    assert len(all_solutions) == 1, "Should generate one container."
    assert isinstance(all_solutions[0], Container), "The generated item should be of type Container."
    assert all_solutions[0].name == "Container1"


def test_generate_with_more_than_one_source(handles_and_containers_world):
    world = handles_and_containers_world

    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
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


def test_sources(handles_and_containers_world):
    world = let("world", type_=World, domain=handles_and_containers_world)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)

    query = an(set_of(drawer_components,
                      container == fixed_connection.parent,
                      handle == fixed_connection.child,
                      container == prismatic_connection.child
                      )
               )
    # render_tree(handle._sources_[0]._node_.root, use_dot_exporter=True, view=True)
    sources = query._sources_
    assert len(sources) == 1, "Should have four sources."
    assert sources[0] == world , "The source should be the world."


def test_the(handles_and_containers_world):
    world = handles_and_containers_world

    with pytest.raises(MultipleSolutionFound):
        handle = the(entity(body := let("body", type_=Handle, domain=world.bodies),
                            body.name.startswith("Handle"))).evaluate()

    handle = the(entity(body := let("body", type_=Handle, domain=world.bodies),
                        body.name.startswith("Handle1"))).evaluate()


def test_not_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle = an(
        entity(body := let("body", type_=Body, domain=world.bodies),
               Not(body.name.startswith("Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_comparator(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle = an(entity(body := let("body", type_=Body, domain=world.bodies),
                           Not(contains(body.name, "Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_and(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      Not(contains(body.name, "Handle") & contains(body.name, '1'))
                      )
               )

    all_not_handle1 = list(query.evaluate())
    assert len(all_not_handle1) == 5, "Should generate 5 bodies"
    assert all(h.name != "Handle1" for h in all_not_handle1), "All generated items should satisfy query"


def test_not_or(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      Not(contains(body.name, "Handle1") | contains(body.name, 'Handle2'))
                      )
               )

    all_not_handle1_or2 = list(query.evaluate())
    assert len(all_not_handle1_or2) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Handle2"] for h in all_not_handle1_or2), "All generated items should satisfy query"


def test_not_and_or(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      Not(Or(And(contains(body.name, "Handle"),
                                 contains(body.name, '1'))
                             , And(contains(body.name, 'Container'),
                                   contains(body.name, '1'))
                             ))
                      )
               )

    all_not_handle1_and_not_container1 = list(query.evaluate())
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"] for h in
        all_not_handle1_and_not_container1), "All generated items should satisfy query"
    print(f"\nCache Search Count = {_cache_search_count.values}")
    print(f"\nCache Match Count = {_cache_match_count.values}")
    # query._render_tree_()


def test_not_and_or_with_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1_and_not_container1 = an(entity(body := let("body", type_=Body, domain=world.bodies),
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
    world = let("world", type_=World, domain=handles_and_containers_world)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
    with SymbolicRule():
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


def test_add_conclusion(handles_and_containers_world):
    world = handles_and_containers_world

    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)

    query = an(entity(drawers := let("drawers", type_=Drawer),
                      And(container == fixed_connection.parent,
                          handle == fixed_connection.child,
                          container == prismatic_connection.child))
               )
    with SymbolicRule(query):
        Add(drawers, Drawer(handle=handle, container=container))

    solutions = query.evaluate()
    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"
    all_drawers = list(drawers)
    assert len(all_drawers) == 2, "Should generate components for two possible drawer."


def test_rule_tree_with_a_refinement(doors_and_drawers_world):
    world = let("world", type_=World, domain=doors_and_drawers_world)
    body = let("body", type_=Body, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)

    query = an(entity(drawers_and_doors := let("drawers_and_doors", type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child))

    with SymbolicRule(query):
        Add(drawers_and_doors, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(drawers_and_doors, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer and 1 door."
    assert isinstance(all_solutions[1], Door)
    assert all_solutions[1].handle.name == "Handle2"
    assert all_solutions[1].body.name == "Body2"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Body4"
    assert isinstance(all_solutions[0], Drawer)
    assert all_solutions[0].handle.name == "Handle1"
    assert all_solutions[0].container.name == "Container1"


def test_rule_tree_with_multiple_refinements(doors_and_drawers_world):
    world = let("world", type_=World, domain=doors_and_drawers_world)
    body = let("body", type_=Body, domain=world.bodies)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    revolute_connection = let("revolute_connection", type_=RevoluteConnection, domain=world.connections)

    query = an(entity(views := let("views", type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child))

    with SymbolicRule(query):
        Add(views, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(views, Door(handle=handle, body=body))
            with alternative(body == revolute_connection.child, container == revolute_connection.parent):
                Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[1], Door)
    assert all_solutions[1].handle.name == "Handle2"
    assert all_solutions[1].body.name == "Body2"
    assert isinstance(all_solutions[2], Wardrobe)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Container2"
    assert all_solutions[2].body.name == "Body4"
    assert isinstance(all_solutions[0], Drawer)
    assert all_solutions[0].handle.name == "Handle1"
    assert all_solutions[0].container.name == "Container1"


def test_rule_tree_with_an_alternative(doors_and_drawers_world):
    world = let("world", type_=World, domain=doors_and_drawers_world)
    body = let("body", type_=Body, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    revolute_connection = let("revolute_connection", type_=RevoluteConnection, domain=world.connections)

    query = an(entity(views := let("views", type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child))

    with SymbolicRule(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(body == revolute_connection.parent, handle == revolute_connection.child):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 4, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle2"
    assert all_solutions[2].container.name == "Body2"
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].body.name == "Body3"
    assert isinstance(all_solutions[3], Drawer)
    assert all_solutions[3].handle.name == "Handle4"
    assert all_solutions[3].container.name == "Body4"
    assert isinstance(all_solutions[1], Drawer)
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"


def test_rule_tree_with_multiple_alternatives(doors_and_drawers_world):
    world = let("world", type_=World, domain=doors_and_drawers_world)
    body = let("body", type_=Body, domain=world.bodies)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let("revolute_connection", type_=RevoluteConnection, domain=world.connections)

    query = an(entity(views := let("views", type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child,
                      body == prismatic_connection.child
                      ))

    with SymbolicRule(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(body == revolute_connection.parent, handle == revolute_connection.child):
            Add(views, Door(handle=handle, body=body))
        with alternative(handle == fixed_connection.child, body == fixed_connection.parent,
                         body == revolute_connection.child,
                         container == revolute_connection.parent):
            Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    print(f"\nCache Enter Count = {_cache_enter_count.values}")
    print(f"\nCache Search Count = {_cache_search_count.values}")
    print(f"\nCache Match Count = {_cache_match_count.values}")
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].body.name == "Body3"
    assert isinstance(all_solutions[2], Wardrobe)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Container2"
    assert all_solutions[2].body.name == "Body4"
    assert isinstance(all_solutions[1], Drawer)
    assert all_solutions[1].container.name == "Container1"
    assert all_solutions[1].handle.name == "Handle1"
    # print(f"\nCache Match Percent = {_cache_match_count.values/_cache_search_count.values}")
