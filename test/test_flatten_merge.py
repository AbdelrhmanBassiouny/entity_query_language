from entity_query_language import a, set_of, symbolic_mode, let, From
from entity_query_language.entity import flatten, entity, an, not_, in_, merge
from .datasets import View, Drawer, Container, Handle, Cabinet


# Make a simple View-like container with an iterable attribute `drawers` to be flattened
class CabinetLike(View):
    def __init__(self, drawers, world):
        super().__init__(world=world)
        self.drawers = list(drawers)


def test_flatten_iterable_attribute(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        views = Cabinet(From(world.views))
        drawers = flatten(views.drawers)
        query = an(entity(drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 3
    assert {row.handle.name for row in results} == {"Handle1", "Handle2", "Handle3"}


def test_flatten_iterable_attribute_and_use_not_equal(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        cabinets = Cabinet(From(world.views))
        drawer_1 = an(entity(d:= Drawer(From(world.views)), d.handle.name == "Handle1"))
        drawers = flatten(cabinets.drawers)
        query = an(entity(drawers, drawer_1 != drawers))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 2
    assert {row.handle.name for row in results} == {"Handle2", "Handle3"}


def test_merge(handles_and_containers_world):
    world = handles_and_containers_world

    with symbolic_mode():
        cabinets = Cabinet(From(world.views))
        my_drawers = an(entity(d := Drawer(From(world.views)), d.handle.name == "Handle1"))
        drawers = merge(cabinets.drawers)
        query = an(entity(my_drawers, not_(in_(my_drawers, drawers))))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 0

    with symbolic_mode():
        cabinets = Cabinet(From(world.views))
        my_drawers = an(entity(d := Drawer(From(world.views)), d.handle.name == "Handle1"))
        drawers = merge(cabinets.drawers)
        query = an(entity(my_drawers, in_(my_drawers, drawers)))

    results = list(query.evaluate())

    # We should get one row for each drawer and the parent view preserved
    assert len(results) == 1
    assert results[0].handle.name == "Handle1"
