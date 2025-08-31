# Example with `Not`

Negation is important and tricky. EQL tries to optimize the query when negation is used which greatly lowers wait time
tof first response. This is done by avoiding evaluating all possibilities to evaluation the negation.

## Example Usage

```python
from entity_query_language import entity, let, an, and_, or_, not_
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
                   not_(and_(or_(body.name.startswith("C"), body.name.endswith("1")),
                            or_(body.name.startswith("H"), body.name.endswith("1"))
                            ))
                   )
            ).evaluate()
results = list(result)
assert len(results) == 2
assert results[0].name == "Container2" and results[1].name == "Handle2"
```

Without the not this example yields `Container1` and `Handle1`, but with the not it yields the complement so
`Container2` and `Handle2`.
