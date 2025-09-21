# Example: Using Cached Symbols Without Explicit Domains

This example is similar to the intro example but intentionally avoids using `From()` or providing a `domain=` inside the query. It demonstrates that all instances of classes decorated with `@symbol` are cached internally in EQL and will be used when the domain is not provided.

```python
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


# Creating instances is enough to populate EQL's internal cache for the type Body
world = World(1, [Body("Body1"), Body("Body2")])

# No domain is provided to let(...). EQL will pull Body instances from its cache.
with symbolic_mode():
    body = let(type_=Body)  # no domain here
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
```
