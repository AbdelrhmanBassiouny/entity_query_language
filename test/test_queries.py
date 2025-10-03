from dataclasses import dataclass

import pytest


from entity_query_language import and_, not_, contains, in_, symbolic_mode, From, predicate, symbol, Predicate
from entity_query_language.cache_data import cache_search_count, cache_match_count, disable_caching
from entity_query_language import an, entity, set_of, let, the, or_, a
from entity_query_language.failures import MultipleSolutionFound
from entity_query_language.predicate import HasType
from .datasets import Handle, Body, Container, FixedConnection, PrismaticConnection, World, Connection


# disable_caching()

def test_empty_conditions(handles_and_containers_world, doors_and_drawers_world):
    world = handles_and_containers_world
    world2 = doors_and_drawers_world
    query = an(entity(body := let(type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_empty_conditions_and_no_domain(handles_and_containers_world, doors_and_drawers_world):
    world = handles_and_containers_world
    world2 = doors_and_drawers_world
    with symbolic_mode():
        query = an(entity(body := let(type_=Body), body.world == world))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_empty_conditions_without_using_entity(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(let(type_=Body, domain=world.bodies))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_reevaluation_of_simple_query(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let(type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."
    assert len(list(query.evaluate())) == len(world.bodies), "Re-eval: Should generate 6 bodies."


def test_empty_conditions_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Body(From(world.bodies))))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_empty_conditions_predicate_form_without_entity(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = a(Body(From(world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_one_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Body(From(world.bodies), name="Handle1")))
    assert len(list(query.evaluate())) == 1, "Should generate 1 body."


def test_one_property_with_an_external_condition_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(connection := Connection(From(world.connections),
                                                   parent=Container(From(world.bodies))),
                          connection.child.name == "Handle1"))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connection."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_one_property_with_an_external_condition_predicate_form_without_entity(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(connection := Connection(From(world.connections), parent=Container(From(world.bodies))),
                   connection.child.name == "Handle1")
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connection."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_nested_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(Connection(From(world.connections), parent=an(entity(Container(From(world.bodies)))),
                                     child=an(entity(Handle(From(world.bodies)))))))
    results = list(query.evaluate())
    assert len(results) == 2, "Should generate 2 connections."
    assert results[0].parent.name == "Container3"
    assert results[0].child.name == "Handle3"
    assert results[1].parent.name == "Container1"
    assert results[1].child.name == "Handle1"


def test_nested_property_predicate_form_without_entity(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = a(Connection(From(world.connections), parent=Container(From(world.bodies)),
                             child=Handle(From(world.bodies)))
                  )
    results = list(query.evaluate())
    assert len(results) == 2, "Should generate 2 connections."
    assert results[0].parent.name == "Container3"
    assert results[0].child.name == "Handle3"
    assert results[1].parent.name == "Container1"
    assert results[1].child.name == "Handle1"


def test_nested_specified_property_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = Connection(From(world.connections), parent=Container(From(world.bodies), name="Container1"),
                           child=Handle(From(world.bodies)))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_filtering_connections_without_joining_with_parent_or_child_queries(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = a(connection := Connection(From(world.connections)), HasType(connection.parent, Container),
                  connection.parent.name == "Container1", HasType(connection.child, Handle))
    # query._render_tree_()
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_nested_specified_property_predicate_form_without_entity_without_domain(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = a(connection := Connection(world=world), HasType(connection.parent, Container),
                  connection.parent.name == "Container1", HasType(connection.child, Handle))
    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_nested_property_with_extra_conditions_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = a(Connection(From(world.connections), parent=a(Container(From(world.bodies))),
                             child=a(handle := Handle(From(world.bodies)), handle.name.endswith('3'))
                             )
                  )
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
        with symbolic_mode():
            yield from a(body := Body(From(world.bodies)), body.name.startswith("Handle")).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "Handle")))

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    with symbolic_mode():
        query = an(entity(body := let(name="body", type_=Body, domain=world.bodies),
                             in_("Handle", body.name)))

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "Handle") & contains(body.name, '1')))

    handles = list(query.evaluate())
    assert len(handles) == 1, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "Handle1") | contains(body.name, 'Handle2')))

    handles = list(query.evaluate())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_multi_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    with symbolic_mode():
        generate_handles_and_container1 = an(entity(body := let(type_=Body, domain=world.bodies),
                             contains(body.name, "Handle1")
                             | contains(body.name, 'Handle2')
                             | contains(body.name, 'Container1')))

    handles_and_container1 = list(generate_handles_and_container1.evaluate())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."


def test_generate_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():
        with symbolic_mode():
            yield from an(entity(body := let(type_=Body, domain=world.bodies),
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
    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
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
        with symbolic_mode():
            query = an(entity(body := let(type_=Body, domain=world.bodies),
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
        with symbolic_mode():
            query = an(entity(body := let(type_=Body, domain=world.bodies),
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

    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
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

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)
    with symbolic_mode():
        solutions = a(set_of(drawer_components,
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


def test_generate_with_more_than_one_source_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        query = a(set_of([container := Container(From(world.bodies)),
                          handle := Handle(From(world.bodies)),
                          prismatic_connection := PrismaticConnection(From(world.connections), child=container),
                          fixed_connection := FixedConnection(From(world.connections), parent=container, child=handle)
                          ]))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert sol[container] == sol[fixed_connection].parent
        assert sol[handle] == sol[fixed_connection].child
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_generate_with_more_than_one_source_optimized(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        q1 = a(fixed_connection:=FixedConnection(From(world.connections)),
          HasType(fixed_connection.parent, Container),
          HasType(fixed_connection.child, Handle))
        q2 = a(prismatic_connection:=PrismaticConnection(From(world.connections), child=fixed_connection.parent))
        query = a(set_of([q1, q2]))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert isinstance(sol[fixed_connection].parent, Container)
        assert isinstance(sol[fixed_connection].child, Handle)
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_sources(handles_and_containers_world):
    with symbolic_mode():
        world = let(type_=World, domain=handles_and_containers_world)
        container = let(type_=Container, domain=world.bodies)
        handle = let(type_=Handle, domain=world.bodies)
        fixed_connection = let(type_=FixedConnection, domain=world.connections)
        prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
        drawer_components = (container, handle, fixed_connection, prismatic_connection)
        query = an(set_of(drawer_components,
                          container == fixed_connection.parent,
                          handle == fixed_connection.child,
                          container == prismatic_connection.child
                          )
                   )
    # render_tree(handle._sources_[0]._node_.root, use_dot_exporter=True, view=True)
    sources = list(query._sources_)
    assert len(sources) == 1, "Should have 1 source."
    assert sources[0].value is handles_and_containers_world, "The source should be the world."


def test_the(handles_and_containers_world):
    world = handles_and_containers_world

    with pytest.raises(MultipleSolutionFound):
        with symbolic_mode():
            handle = the(entity(body := let(type_=Handle, domain=world.bodies),
                                body.name.startswith("Handle"))).evaluate()
    with symbolic_mode():
        handle = the(entity(body := let(type_=Handle, domain=world.bodies),
                            body.name.startswith("Handle1"))).evaluate()


def test_not_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        not_handle = an(
            entity(body := let(type_=Body, domain=world.bodies),
                   not_(body.name.startswith("Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_comparator(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        not_handle = an(entity(body := let(type_=Body, domain=world.bodies),
                               not_(contains(body.name, "Handle")))).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_and(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                          not_(contains(body.name, "Handle") & contains(body.name, '1'))
                          )
                   )

    all_not_handle1 = list(query.evaluate())
    assert len(all_not_handle1) == 5, "Should generate 5 bodies"
    assert all(h.name != "Handle1" for h in all_not_handle1), "All generated items should satisfy query"


def test_not_or(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                          not_(contains(body.name, "Handle1") | contains(body.name, 'Handle2'))
                          )
                   )

    all_not_handle1_or2 = list(query.evaluate())
    assert len(all_not_handle1_or2) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Handle2"] for h in all_not_handle1_or2), "All generated items should satisfy query"


def test_not_and_or(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
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


def test_empty_list_literal(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                          not_(contains([], "Handle") & contains(body.name, '1'))
                          )
                   )
    results = list(query.evaluate())


def test_not_and_or_with_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        not_handle1_and_not_container1 = an(entity(body := let(type_=Body, domain=world.bodies),
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


def test_generate_with_using_decorated_predicate(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    @predicate
    def is_handle(body_: Body):
        return body_.name.startswith("Handle")

    with symbolic_mode():
        query = an(entity(body := let(type_=Body, domain=world.bodies),
                          is_handle(body_=body)))

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."


def test_generate_with_using_inherited_predicate(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    @dataclass(frozen=True)
    class IsHandle(Predicate):
        body: Body

        def __call__(self):
            return self.body.name.startswith("Handle")

    with symbolic_mode():
        query = an(entity(body := Body(From(world.bodies)), IsHandle(body=body)))

    handles = list(query.evaluate())

    assert len(handles) == 3, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in handles), "All generated items should be of type Handle."
    assert all(IsHandle(h)() for h in handles), "All generated items should satisfy the predicate."
    # assert all(not IsHandle(b).should_infer for b in world.bodies), ("All seen items should not be inferred again"
    #                                                                  " but retrieved.")
    assert all(not IsHandle(b)() for b in world.bodies if b not in handles), ("All not generated items "
                                                                              "should not satisfy the "
                                                                              "predicate.")


def test_nested_query_with_or(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        original_query = an(entity(body := let(type_=Body, domain=world.bodies),
                                   contains(body.name, "Handle1") | contains(body.name, 'Handle2')))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 2, "Should generate at least one handle."
    assert all(isinstance(h, Handle) for h in original_query_handles), "All generated items should be of type Handle."

    with symbolic_mode():
        query_part1 = an(entity(body, contains(body.name, "Handle1")))
        query_part2 = an(entity(body, contains(body.name, "Handle2")))
        nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_and(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        original_query = the(entity(body := let(type_=Body, domain=world.bodies),
                                    contains(body.name, "Handle") & contains(body.name, '1')))

    original_query_handle = original_query.evaluate()
    assert original_query_handle.name == "Handle1"

    with symbolic_mode():
        query_part1 = an(entity(body, contains(body.name, "Handle")))
        query_part2 = an(entity(body, contains(body.name, "1")))
        nested_query = the(entity(body, query_part1 & query_part2))

    # nested_query._render_tree_()

    nested_query_handle = nested_query.evaluate()
    assert nested_query_handle == original_query_handle, "Should generate same results"


def test_nested_query_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        original_query = an(entity(body := let(type_=Body, domain=world.bodies),
                                   or_(contains(body.name, "Handle") & contains(body.name, '1'),
                                       contains(body.name, "Handle") & contains(body.name, '2'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 2, "Should generate two handles"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"

    with symbolic_mode():
        query_part1 = an(entity(body, contains(body.name, "Handle") & contains(body.name, '1')))
        query_part2 = an(entity(body, contains(body.name, "Handle") & contains(body.name, '2')))
        nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        original_query = an(entity(body := let(type_=Body, domain=world.bodies),
                                   and_(contains(body.name, "Handle") | contains(body.name, '1'),
                                        contains(body.name, "Handle") | contains(body.name, '2'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 3, "Should generate 3 handles"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"
    assert original_query_handles[2].name == "Handle3"

    with symbolic_mode():
        query_part1 = an(entity(body, contains(body.name, "Handle") | contains(body.name, '1')))
        query_part2 = an(entity(body, contains(body.name, "Handle") | contains(body.name, '2')))
        nested_query = an(entity(body, query_part1 & query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_multi_or(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        original_query = an(entity(body := let(type_=Body, domain=world.bodies),
                                   or_(contains(body.name, "Handle1") | contains(body.name, 'Handle2'),
                                       contains(body.name, 'Container1'))))

    original_query_handles = list(original_query.evaluate())
    assert len(original_query_handles) == 3, "Should generate 2 handles and 1 container"
    assert original_query_handles[0].name == "Handle1"
    assert original_query_handles[1].name == "Handle2"
    assert original_query_handles[2].name == "Container1"

    with symbolic_mode():
        query_part1 = an(entity(body, contains(body.name, "Handle1") | contains(body.name, 'Handle2')))
        query_part2 = an(entity(body, contains(body.name, 'Container1')))
        nested_query = an(entity(body, query_part1 | query_part2))

    # nested_query._render_tree_()

    nested_query_handles = list(nested_query.evaluate())
    assert nested_query_handles == original_query_handles, "Should generate same results"


def test_nested_query_with_multiple_sources(handles_and_containers_world):
    world = handles_and_containers_world
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)

    with symbolic_mode():
        original_query = an(set_of(drawer_components,
                                   container == fixed_connection.parent,
                                   handle == fixed_connection.child,
                                   container == prismatic_connection.child
                                   )
                            )
    # original_query._render_tree_()
    original_query_results = list(original_query.evaluate())
    assert len(original_query_results) == 2, "Should generate 2 drawer components"

    with symbolic_mode():
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
    assert all(nested_res[k] == original_res[k]
               for nested_res, original_res in zip(original_query_results, nested_query_results)
               for k in drawer_components), "Should generate same results"


def test_implicitly_bound_first_predicate_argument(handles_and_containers_world):
    world = handles_and_containers_world
    with symbolic_mode():
        with a(Body(From(world.bodies))) as q:
            HasType(Handle)
    results = list(q.evaluate())
    assert len(results) == 3, "Should generate 3 handles."