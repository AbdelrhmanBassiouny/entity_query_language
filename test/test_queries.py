import pytest

from entity_query_language import and_, not_, contains, in_, symbolic_mode
from entity_query_language.cache_data import cache_search_count, cache_match_count, disable_caching
from entity_query_language.entity import an, entity, set_of, let, the, or_, predicate
from entity_query_language.failures import MultipleSolutionFound
from .datasets import Handle, Body, Container, FixedConnection, PrismaticConnection, World, Connection


# disable_caching()


def test_empty_conditions(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_reevaluation_of_simple_query(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."
    assert len(list(query.evaluate())) == len(world.bodies), "Re-eval: Should generate 6 bodies."



def test_empty_conditions_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := Body(), domain=world.bodies))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_one_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := Body(name="Handle1"), domain=world.bodies))
    assert len(list(query.evaluate())) == 1, "Should generate 1 body."


def test_one_property_with_an_external_condition_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(connection := Connection(parent=let("c1", type_=Container, domain=world.bodies)),
                          connection.child.name == "Handle1",
                          domain=world.connections))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connection."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"

def test_nested_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Connection(parent=an(entity(Container(), domain=world.bodies)),
                                     child=an(entity(Handle(), domain=world.bodies))),
                          domain=world.connections))
    results = list(query.evaluate())
    assert len(results) == 2, "Should generate 2 connections."
    assert results[0].parent.name == "Container3"
    assert results[0].child.name == "Handle3"
    assert results[1].parent.name == "Container1"
    assert results[1].child.name == "Handle1"

def test_nested_specified_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Connection(parent=an(entity(Container(name="Container1"), domain=world.bodies)),
                                     child=an(entity(Handle(), domain=world.bodies))),
                          domain=world.connections))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"

def test_nested_property_with_extra_conditions_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Connection(parent=an(entity(Container(), domain=world.bodies)),
                                     child=an(entity(handle:=Handle(), domain=world.bodies))),
                          handle.name.endswith('3'),
                          domain=world.connections))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container3"
    assert results[0].child.name == "Handle3"

def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():
        yield from an(
            entity(body := let("body", type_=Body, domain=world.bodies),
                   body.name.startswith("Handle"))).evaluate()

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
                             or_(and_(contains(body.name, "Handle"),
                                      contains(body.name, '1'))
                                 , and_(contains(body.name, 'Container'),
                                        contains(body.name, '1'))
                                 )
                             )
                      ).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_reevaluation_of_or_and_query(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                         or_(and_(contains(body.name, "Handle"),
                                  contains(body.name, '1'))
                             , and_(contains(body.name, 'Container'),
                                    contains(body.name, '1'))
                             )
                         )
                  )

    handles_and_container1 = list(query.evaluate())
    assert len(handles_and_container1) == 2, "Should generate one handle and one container."
    handles_and_container1 = list(query.evaluate())
    assert len(handles_and_container1) == 2, "Re-eval: Should generate one handle and one container."


def test_generate_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                          or_(contains(body.name, "Handle"), contains(body.name, '1'))
                          , or_(contains(body.name, 'Container'), contains(body.name, '1'))
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


def test_reevaluate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      contains(body.name, "n"), contains(body.name, '1')
                      , contains(body.name, 'C')))

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 1, "Should generate one container."
    assert isinstance(all_solutions[0], Container), "The generated item should be of type Container."
    assert all_solutions[0].name == "Container1"
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 1, "Re-eval: Should generate one container."
    assert isinstance(all_solutions[0], Container), "Re-eval: The generated item should be of type Container."
    assert all_solutions[0].name == "Container1", "Re-eval: The generated item should be of type Container."

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
    assert sources[0] == world, "The source should be the world."


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
               not_(body.name.startswith("Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_comparator(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle = an(entity(body := let("body", type_=Body, domain=world.bodies),
                           not_(contains(body.name, "Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_and(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      not_(contains(body.name, "Handle") & contains(body.name, '1'))
                      )
               )

    all_not_handle1 = list(query.evaluate())
    assert len(all_not_handle1) == 5, "Should generate 5 bodies"
    assert all(h.name != "Handle1" for h in all_not_handle1), "All generated items should satisfy query"


def test_not_or(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      not_(contains(body.name, "Handle1") | contains(body.name, 'Handle2'))
                      )
               )

    all_not_handle1_or2 = list(query.evaluate())
    assert len(all_not_handle1_or2) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Handle2"] for h in all_not_handle1_or2), "All generated items should satisfy query"


def test_not_and_or(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                      not_(or_(and_(contains(body.name, "Handle"),
                                    contains(body.name, '1'))
                               , and_(contains(body.name, 'Container'),
                                      contains(body.name, '1'))
                               ))
                      )
               )

    all_not_handle1_and_not_container1 = list(query.evaluate())
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"] for h in
        all_not_handle1_and_not_container1), "All generated items should satisfy query"
    print(f"\nCache Search Count = {cache_search_count.values}")
    print(f"\nCache Match Count = {cache_match_count.values}")
    # query._render_tree_()


def test_not_and_or_with_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    not_handle1_and_not_container1 = an(entity(body := let("body", type_=Body, domain=world.bodies),
                                               not_(and_(or_(body.name.startswith("Handle"),
                                                             body.name.endswith('1'))
                                                         , or_(body.name.startswith('Container'),
                                                               body.name.endswith('1'))
                                                         ))
                                               )
                                        ).evaluate()

    all_not_handle1_and_not_container1 = list(not_handle1_and_not_container1)
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"] for h in
        all_not_handle1_and_not_container1), "All generated items should satisfy query"


def test_generate_with_using_predicate(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    @predicate
    def is_handle(body_: Body):
        return body_.name.startswith("Handle")

    with symbolic_mode():
        query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                          is_handle(body_=body)))

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_nested_query_with_or(handles_and_containers_world):
    world = handles_and_containers_world

    original_query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                               contains(body.name, "Handle1") | contains(body.name, 'Handle2')))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in original_query_handles), "All generated items should be of type Handle."

    query_part1 = an(entity(body, contains(body.name, "Handle1")))
    query_part2 = an(entity(body, contains(body.name, "Handle2")))
    nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_and(handles_and_containers_world):
    world = handles_and_containers_world

    original_query = the(entity(body := let("body", type_=Body, domain=world.bodies),
                                contains(body.name, "Handle") & contains(body.name, '1')))

    original_query_handle = original_query.evaluate()
    assert original_query_handle.name == "Handle1"

    query_part1 = an(entity(body, contains(body.name, "Handle")))
    query_part2 = an(entity(body, contains(body.name, "1")))
    nested_query = the(entity(body, query_part1 & query_part2))

    # nested_query._render_tree_()

    nested_query_handle = nested_query.evaluate()
    assert nested_query_handle == original_query_handle, "Should generate same results"


def test_nested_query_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world

    original_query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                               or_(contains(body.name, "Handle") & contains(body.name, '1'),
                                   contains(body.name, "Handle") & contains(body.name, '2'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 2, "Should generate two handles"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"

    query_part1 = an(entity(body, contains(body.name, "Handle") & contains(body.name, '1')))
    query_part2 = an(entity(body, contains(body.name, "Handle") & contains(body.name, '2')))
    nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    original_query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                               and_(contains(body.name, "Handle") | contains(body.name, '1'),
                                    contains(body.name, "Handle") | contains(body.name, '2'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 3, "Should generate 3 handles"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"
    assert original_query_handles[2].name == "Handle3"

    query_part1 = an(entity(body, contains(body.name, "Handle") | contains(body.name, '1')))
    query_part2 = an(entity(body, contains(body.name, "Handle") | contains(body.name, '2')))
    nested_query = an(entity(body, query_part1 & query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_multi_or(handles_and_containers_world):
    world = handles_and_containers_world

    original_query = an(entity(body := let("body", type_=Body, domain=world.bodies),
                               or_(contains(body.name, "Handle1") | contains(body.name, 'Handle2'),
                                   contains(body.name, 'Container1'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 3, "Should generate 2 handles and 1 container"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"
    assert original_query_handles[2].name == "Container1"

    query_part1 = an(entity(body, contains(body.name, "Handle1") | contains(body.name, 'Handle2')))
    query_part2 = an(entity(body, contains(body.name, 'Container1')))
    nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_multiple_sources(handles_and_containers_world):
    world = let("world", type_=World, domain=handles_and_containers_world)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)

    original_query = an(set_of(drawer_components,
                               container == fixed_connection.parent,
                               handle == fixed_connection.child,
                               container == prismatic_connection.child
                               )
                        )

    original_query_results = list(original_query.evaluate())
    assert len(original_query_results) == 2, "Should generate 2 drawer components"

    query1 = an(set_of((container, fixed_connection),
                       container == fixed_connection.parent
                       )
                )
    query2 = an(set_of(drawer_components, handle == fixed_connection.child,
                       container == prismatic_connection.child,
                       ))
    nested_query = an(set_of(drawer_components, query1 & query2))

    nested_query_results = list(nested_query.evaluate())
    assert len(nested_query_results) == 2, "Should generate 2 drawer components"
    assert nested_query_results == original_query_results, "Should generate same results"
