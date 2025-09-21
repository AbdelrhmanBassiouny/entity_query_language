# Example: Using indexing in symbolic expressions

This example demonstrates capturing indexing (the __getitem__ operator) on symbolic variables. You can index into
containers held by your symbolic variable, and the operation will be represented symbolically in the query.

```python
from dataclasses import dataclass
from typing_extensions import List, Dict
from entity_query_language import an, entity, let, symbolic_mode, symbol

@symbol
@dataclass
class Body:
    name: str
    props: Dict[str, int]

@dataclass
class World:
    bodies: List[Body]

world = World([
    Body("Body1", {"score": 1}),
    Body("Body2", {"score": 2}),
])

with symbolic_mode():
    b = let(type_=Body, domain=world.bodies)
    # use indexing on a dict field
    query = an(entity(b, b.props["score"] == 2))

results = list(query.evaluate())
assert len(results) == 1 and results[0].name == "Body2"
```
