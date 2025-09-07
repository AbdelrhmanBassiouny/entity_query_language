from examples.intro_example import prismatic_connection
from .datasets import World, Container, Handle, FixedConnection, PrismaticConnection, Drawer, View, Door, Body, \
    RevoluteConnection, Wardrobe
from entity_query_language import let, symbolic_mode, an, entity, and_
from entity_query_language.rule import refinement, alternative
from entity_query_language.conclusion import Add
from entity_query_language.cache_data import cache_enter_count, cache_search_count, cache_match_count, \
    cache_lookup_time, cache_update_time


def test_generate_drawers(handles_and_containers_world):
    world = let("world", type_=World, domain=handles_and_containers_world)
    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)
    with symbolic_mode():
        solutions = an(entity(Drawer(handle=handle, container=container),
                              and_(container == fixed_connection.parent,
                                   handle == fixed_connection.child,
                                   container == prismatic_connection.child))).evaluate()

    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].container.name == "Container3"
    assert all_solutions[0].handle.name == "Handle1"
    assert all_solutions[0].container.name == "Container1"


def test_generate_drawers_predicate_form(handles_and_containers_world):
    world = let("world", type_=World, domain=handles_and_containers_world)
    with symbolic_mode():
        query = an(entity(Drawer(handle=an(entity(handle:=Handle(), domain=world.bodies)),
                                 container=an(entity(container:=Container(), domain=world.bodies))),
                          an(entity(FixedConnection(parent=container, child=handle), domain=world.connections)),
                          an(entity(PrismaticConnection(child=container), domain=world.connections))))

    # query._render_tree_()
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].container.name == "Container3"
    assert all_solutions[0].handle.name == "Handle1"
    assert all_solutions[0].container.name == "Container1"


def test_add_conclusion(handles_and_containers_world):
    world = handles_and_containers_world

    container = let("container", type_=Container, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    prismatic_connection = let("prismatic_connection", type_=PrismaticConnection, domain=world.connections)

    query = an(entity(drawers := let("drawers", type_=Drawer),
                      and_(container == fixed_connection.parent,
                           handle == fixed_connection.child,
                           container == prismatic_connection.child))
               )
    with symbolic_mode(query):
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

    with symbolic_mode(query):
        Add(drawers_and_doors, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(drawers_and_doors, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer and 1 door."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Drawer)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


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

    with symbolic_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(views, Door(handle=handle, body=body))
            with alternative(body == revolute_connection.child, container == revolute_connection.parent):
                Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Wardrobe)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Container2"
    assert all_solutions[1].body.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


def test_rule_tree_with_an_alternative(doors_and_drawers_world):
    world = let("world", type_=World, domain=doors_and_drawers_world)
    body = let("body", type_=Body, domain=world.bodies)
    handle = let("handle", type_=Handle, domain=world.bodies)
    fixed_connection = let("fixed_connection", type_=FixedConnection, domain=world.connections)
    revolute_connection = let("revolute_connection", type_=RevoluteConnection, domain=world.connections)

    query = an(entity(views := let("views", type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child))

    with symbolic_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(body == revolute_connection.parent, handle == revolute_connection.child):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 4, "Should generate 2 drawers, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Drawer)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].container.name == "Body2"
    assert isinstance(all_solutions[1], Door)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].body.name == "Body3"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Body4"
    assert isinstance(all_solutions[3], Drawer)
    assert all_solutions[3].handle.name == "Handle1"
    assert all_solutions[3].container.name == "Container1"


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

    with symbolic_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(body == revolute_connection.parent, handle == revolute_connection.child):
            Add(views, Door(handle=handle, body=body))
        with alternative(handle == fixed_connection.child, body == fixed_connection.parent,
                         body == revolute_connection.child,
                         container == revolute_connection.parent):
            Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()
    all_solutions = list(query.evaluate())
    print(f"\nCache Enter Count = {cache_enter_count.values}")
    print(f"\nCache Search Count = {cache_search_count.values}")
    print(f"\nCache Match Count = {cache_match_count.values}")
    print(f"\nCache LookUp Time = {cache_lookup_time.values}")
    print(f"\nCache Update Time = {cache_update_time.values}")
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].body.name == "Body3"
    assert isinstance(all_solutions[1], Wardrobe)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Container2"
    assert all_solutions[1].body.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].container.name == "Container1"
    assert all_solutions[2].handle.name == "Handle1"
    # print(f"\nCache Match Percent = {_cache_match_count.values/_cache_search_count.values}")
