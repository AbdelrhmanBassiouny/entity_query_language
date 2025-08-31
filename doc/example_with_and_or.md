# Example with `And` + `Or`

Here is an example of a more nested query conditions.

## Example Usage

```python
from entity_query_language import entity, let, an, and_, or_
from dataclasses import dataclass
from typing_extensions import List


@dataclass(unsafe_hash=True)
class Body:
    name: str


@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
result = an(entity(body := let("body", type_=Body, domain=world.bodies),
                   and_(or_(body.name.startswith("C"), body.name.endswith("1")),
                        or_(body.name.startswith("H"), body.name.endswith("1"))
                        )
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"
```

This way of writing `And`, `Or` is exactly like constructing a tree which allows for the user to write in the same
structure as how the computation is done internally.
