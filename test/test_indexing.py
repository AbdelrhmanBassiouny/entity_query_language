from dataclasses import dataclass
from typing_extensions import Dict, List

from entity_query_language import an, entity, let, symbolic_mode, symbol


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
