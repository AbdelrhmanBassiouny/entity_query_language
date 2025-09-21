from dataclasses import dataclass
from typing_extensions import Dict, List

from entity_query_language import an, entity, let, symbolic_mode, symbol, From


def test_indexing_on_dict_field():
    @symbol
    @dataclass(unsafe_hash=True)
    class Item:
        name: str
        attrs: Dict[str, int]

    @dataclass(eq=False)
    class World:
        items: List[Item]

    world = World([
        Item("A", {"score": 1}),
        Item("B", {"score": 2}),
        Item("C", {"score": 2}),
    ])

    with symbolic_mode():
        i = let(type_=Item, domain=world.items)
        q = an(entity(i, i.attrs["score"] == 2))
    res = list(q.evaluate())
    assert {x.name for x in res} == {"B", "C"}


def test_indexing_2():
    @symbol
    @dataclass
    class Shape:
        name: str
        color: str

    @symbol
    @dataclass
    class Body:
        shapes: List[Shape]

    world_bodies = [Body(shapes=[Shape('shape1', color='red'), Shape('shape2', color='blue')]),
                                 Body(shapes=[Shape('shape1', color='green'), Shape('shape2', color='black')])]
    with symbolic_mode():
        body = Body(From(world_bodies))
        body_tha_has_red_shape = an(entity(body, body.shapes[0].color == 'red')).evaluate()
    body_tha_has_red_shape = list(body_tha_has_red_shape)
    assert len(body_tha_has_red_shape) == 1
    assert body_tha_has_red_shape[0].shapes[0].color == 'red'