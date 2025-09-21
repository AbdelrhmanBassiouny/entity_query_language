from dataclasses import dataclass

from typing_extensions import List

from entity_query_language import entity, an, let, contains, symbolic_mode, symbol


@symbol
@dataclass
class Body:
    name: str


@dataclass
class World:
    id_: int
    bodies: List[Body]


# All instances of classes decorated with @symbol are cached internally in EQL.
# This means that once you create them, you can reference their type in queries
# without explicitly specifying a domain (From()/domain=). EQL will pull values
# for that type from its internal cache when possible.
world = World(1, [Body("Body1"), Body("Body2")])

# Build the cache by just creating instances above. Now write a query that
# does NOT pass a domain to let(...). EQL will use cached Body instances.
with symbolic_mode():
    body = let(type_=Body)  # no domain provided here on purpose
    query = an(
        entity(
            body,
            contains(body.name, "2"),
            body.name.startswith("Body"),
        )
    )

results = list(query.evaluate())
assert len(results) == 1
assert results[0].name == "Body2"
