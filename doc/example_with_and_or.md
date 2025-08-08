# Example with `And` + `Or`

Here is an example of a more nested query conditions.

## Example Usage

```python
from entity_query_language import entity, let, an, And, Or
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
result = an(entity(body := let(type_=Body, domain=world.bodies),
                   And(Or(body.name.startswith("C"), body.name.endswith("1")),
                       Or(body.name.startswith("H"), body.name.endswith("1"))
                       )
                   )
            )
results = list(result)
assert len(results) == 2
assert results[0].name == "Container1" and results[1].name == "Handle1"
```

`body1` will execute successfully giving one solution wich is the body with the name `Body1`.
`body` will raise an error is there is multiple bodies which have a name that starts with `Body`.
