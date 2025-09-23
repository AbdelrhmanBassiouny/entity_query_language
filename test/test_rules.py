from entity_query_language import let, an, entity, and_, a
from entity_query_language.cache_data import cache_enter_count, cache_search_count, cache_match_count, \
    cache_lookup_time, cache_update_time
from entity_query_language.conclusion import Add
from entity_query_language.entity import infer
from entity_query_language.predicate import HasType
from entity_query_language.rule import refinement, alternative, next_rule
from entity_query_language.symbolic import rule_mode, From, symbolic_mode
from .datasets import Container, Handle, FixedConnection, PrismaticConnection, Drawer, View, Door, Body, \
    RevoluteConnection, Wardrobe


def test_generate_drawers(handles_and_containers_world):
    world = handles_and_containers_world
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    with rule_mode():
        solutions = infer(Drawer(handle=handle, container=container),
                          and_(container == fixed_connection.parent,
                               handle == fixed_connection.child,
                               container == prismatic_connection.child)).evaluate()

    all_solutions = list(solutions)

    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"


def test_generate_drawers_predicate_form(handles_and_containers_world):
    world = handles_and_containers_world
    with rule_mode():
        query = an(entity(Drawer(handle=an(entity(handle := Handle(From(world.bodies)))),
                                 container=an(entity(container := Container(From(world.bodies))))),
                          an(entity(FixedConnection(From(world.connections), parent=container, child=handle))),
                          an(entity(PrismaticConnection(From(world.connections), child=container)))))

    # query._render_tree_()
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"


def test_generate_drawers_predicate_form_without_entity(handles_and_containers_world):
    world = handles_and_containers_world
    with rule_mode():
        query = a(Drawer(handle=a(handle := Handle(From(world.bodies))),
                         container=a(container := Container(From(world.bodies)))),
                  PrismaticConnection(From(world.connections), child=container),
                  FixedConnection(From(world.connections), parent=container, child=handle))

    # query._render_tree_()
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].container.name == "Container3"
    assert all_solutions[0].handle.name == "Handle1"
    assert all_solutions[0].container.name == "Container1"


def test_generate_drawers_predicate_form_without_entity_and_domain(handles_and_containers_world):
    world = handles_and_containers_world
    with rule_mode():
        query = infer(Drawer(handle=a(handle := Handle(world=world)),
                             container=a(container := Container(world=world))),
                      PrismaticConnection(child=container, world=world),
                      FixedConnection(parent=container, child=handle, world=world))

    # query._render_tree_()
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    expected_name_tuple_set = {("Handle3", "Container3"), ("Handle1", "Container1")}
    solution_name_tuple_set = {(s.handle.name, s.container.name) for s in all_solutions}
    assert expected_name_tuple_set == solution_name_tuple_set


def test_add_conclusion(handles_and_containers_world):
    world = handles_and_containers_world

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)

    with symbolic_mode():
        query = an(entity(drawers := let(type_=Drawer),
                          container == fixed_connection.parent,
                          handle == fixed_connection.child,
                          container == prismatic_connection.child)
                   )
    with rule_mode(query):
        Add(drawers, Drawer(handle=handle, container=container))

    solutions = query.evaluate()
    all_solutions = list(solutions)
    assert len(all_solutions) == 2, "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"
    # all_drawers = list(drawers._evaluate__())
    # assert len(all_drawers) == 2, "Should generate components for two possible drawer."


def test_rule_tree_with_a_refinement(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)

    with symbolic_mode():
        query = an(entity(drawers_and_doors := let(type_=View),
                          body == fixed_connection.parent,
                          handle == fixed_connection.child))

    with rule_mode(query):
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
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = an(entity(views := let(type_=View),
                          body == fixed_connection.parent,
                          handle == fixed_connection.child))

    with rule_mode(query):
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
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = an(entity(views := let(type_=View),
                          body == fixed_connection.parent,
                          handle == fixed_connection.child))

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(body == revolute_connection.parent, handle == revolute_connection.child):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 4, "Should generate 3 drawers, 1 door"
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
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(views := let(type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child,
                      body == prismatic_connection.child)

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(revolute_connection.parent == body, revolute_connection.child == handle):
            Add(views, Door(handle=handle, body=body))
        with alternative(fixed_connection.parent == body, fixed_connection.child == handle,
                         body == revolute_connection.child,
                         container == revolute_connection.parent):
            Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_optimized(doors_and_drawers_world):
    world = doors_and_drawers_world
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(views := let(type_=View),
                      HasType(fixed_connection.child, Handle),
                      fixed_connection.parent == prismatic_connection.child)

    with rule_mode(query):
        Add(views, Drawer(handle=fixed_connection.child, container=fixed_connection.parent))
        with alternative(HasType(revolute_connection.child, Handle)):
            Add(views, Door(handle=revolute_connection.child, body=revolute_connection.parent))
        with alternative(fixed_connection,
                         fixed_connection.parent == revolute_connection.child,
                         HasType(revolute_connection.parent, Container)):
            Add(views, Wardrobe(handle=fixed_connection.child, body=fixed_connection.parent,
                                container=revolute_connection.parent))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(views := let(type_=View),
                      body == fixed_connection.parent,
                      handle == fixed_connection.child)

    with rule_mode(query):
        with refinement(prismatic_connection.child == body):
            Add(views, Drawer(handle=handle, container=body))
            with alternative(body == revolute_connection.child,
                             container == revolute_connection.parent):
                Add(views, Wardrobe(handle=handle, body=body, container=container))
        with alternative(revolute_connection.parent == body, revolute_connection.child == handle):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree_optimized(doors_and_drawers_world):
    world = doors_and_drawers_world
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(views := let(type_=View),
                      HasType(fixed_connection.child, Handle))

    with rule_mode(query):
        with refinement(prismatic_connection.child == fixed_connection.parent):
            Add(views, Drawer(handle=fixed_connection.child, container=fixed_connection.parent))
            with alternative(fixed_connection.parent == revolute_connection.child,
                             HasType(revolute_connection.parent, Container)):
                Add(views, Wardrobe(handle=fixed_connection.child, body=fixed_connection.parent,
                                    container=revolute_connection.parent))
        with next_rule(HasType(revolute_connection.child, Handle)):
            Add(views, Door(handle=revolute_connection.child, body=revolute_connection.parent))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_predicate_form_too_much_joins(doors_and_drawers_world):
    world = doors_and_drawers_world
    with symbolic_mode():
        body = Body(world=world)
        handle = Handle(world=world)
        container = Container(world=world)
        fixed_connection = FixedConnection(parent=body, child=handle, world=world)
        prismatic_connection = PrismaticConnection(child=body, world=world)
        revolute_connection = RevoluteConnection(parent=body, child=handle, world=world)
        query = infer(views := View(), fixed_connection, prismatic_connection)

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body, world=world))
        with alternative(revolute_connection):
            Add(views, Door(handle=handle, body=body, world=world))
        with alternative(fixed_connection,
                         body == revolute_connection.child,
                         container == revolute_connection.parent,
                         revolute_connection.world == world):
            Add(views, Wardrobe(handle=handle, body=body, container=container, world=world))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    print(f"\nCache Enter Count = {cache_enter_count.values}")
    print(f"\nCache Search Count = {cache_search_count.values}")
    print(f"\nCache Match Count = {cache_match_count.values}")
    print(f"\nCache LookUp Time = {cache_lookup_time.values}")
    print(f"\nCache Update Time = {cache_update_time.values}")
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_predicate_form_better_rule_tree(doors_and_drawers_world):
    world = doors_and_drawers_world
    with symbolic_mode():
        body = Body(world=world)
        handle = Handle(world=world)
        container = Container(world=world)
        fixed_connection = FixedConnection(parent=body, child=handle, world=world)
        prismatic_connection = PrismaticConnection(child=body, world=world)
        revolute_connection = RevoluteConnection(world=world)
        query = infer(views := View(), fixed_connection)

    with rule_mode(query):
        with refinement(prismatic_connection):
            Add(views, Drawer(handle=handle, container=body, world=world))
            with alternative(body == revolute_connection.child,
                             container == revolute_connection.parent,
                             revolute_connection.world == world):
                Add(views, Wardrobe(handle=handle, body=body, container=container, world=world))
        with next_rule(revolute_connection.parent == body,
                       revolute_connection.child == handle):
            Add(views, Door(handle=handle, body=body, world=world))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    print(f"\nCache Enter Count = {cache_enter_count.values}")
    print(f"\nCache Search Count = {cache_search_count.values}")
    print(f"\nCache Match Count = {cache_match_count.values}")
    print(f"\nCache LookUp Time = {cache_lookup_time.values}")
    print(f"\nCache Update Time = {cache_update_time.values}")
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_predicate_form_better_rule_tree_optimized(doors_and_drawers_world):
    world = doors_and_drawers_world
    with symbolic_mode():
        fixed_connection = FixedConnection(world=world)
        prismatic_connection = PrismaticConnection(world=world)
        revolute_connection = RevoluteConnection(world=world)
        query = infer(views := View(), HasType(fixed_connection.child, Handle))

    with rule_mode(query):
        with refinement(prismatic_connection.child == fixed_connection.parent):
            Add(views, Drawer(handle=fixed_connection.child, container=fixed_connection.parent, world=world))
            with alternative(HasType(revolute_connection.parent, Container),
                             revolute_connection.child == fixed_connection.parent):
                Add(views, Wardrobe(handle=fixed_connection.child, body=fixed_connection.parent,
                                    container=revolute_connection.parent, world=world))
        with next_rule(HasType(revolute_connection.child, Handle)):
            Add(views, Door(handle=revolute_connection.child, body=revolute_connection.parent, world=world))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    print(f"\nCache Enter Count = {cache_enter_count.values}")
    print(f"\nCache Search Count = {cache_search_count.values}")
    print(f"\nCache Match Count = {cache_match_count.values}")
    print(f"\nCache LookUp Time = {cache_lookup_time.values}")
    print(f"\nCache Update Time = {cache_update_time.values}")
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {(Door, "Handle3", "Body3"), (Drawer, "Handle1", "Container1"),
                             (Wardrobe, "Handle4", "Body4", "Container2")}
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set
