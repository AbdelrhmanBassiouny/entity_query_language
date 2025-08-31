# Entity Query Language


Welcome to the Entity Query Language package!

EQL is a relational query language that is pythonic, and intuitive.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.


## Example Usage

# Basic Example
An important feature of EQL is that you do not need to do operations like JOIN in SQL, this is performed implicitly.
EQL tries to mirror your intent in a query statement with as less boiler plate code as possiple.
For example an attribute access with and equal check to another value is just how you expect:

```python
from entity_query_language import entity, an, let, and_, contains
from dataclasses import dataclass
from typing_extensions import List


@dataclass(unsafe_hash=True)
class Body:
    name: str


@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]


world = World(1, [Body("Body1"), Body("Body2")])

results_generator = an(entity(body := let("body", type_=Body, domain=world.bodies),
                              and_(contains(body.name, "2"), body.name.startswith("Body")))
                       ).evaluate()
results = list(results_generator)
assert len(results) == 1
assert results[0].name == "Body2"
```

where this creates a body variable that gets its values from world.bodies, and filters them to have their att "name"
equal to "Body1".ample shows generic usage of the Ripple Down Rules to classify objects in a propositional setting.

Notice that it is possible to use both provided helper methods by EQL and other methods that are accessible through your
object instance and use them as part of the query conditions.

## More Example Usage

- [Example with `the`](example_with_the.md): This example shows how to use `the` entity wrapper instead of `an`
- [Example with `And` + `OR`](example_with_and_or.md): This shows an example of using `And` with `Or` together.
- [Example with `Not`](example_with_not.md): This shows an example of using `Not`.
- [Example with Joining Multiple Sources](example_with_joining_multiple_sources.md): This shows an example of using and joining multiple sources in your query. 
- [Example with Inference](example_with_inference.md): This shows an example of writing inference rules in EQL.

## To Cite:

```bib
@software{bassiouny2025eql,
author = {Bassiouny, Abdelrhman},
title = {Entity-Query-Language},
url = {https://github.com/AbdelrhmanBassiouny/entity_query_language},
version = {1.0.0},
}
```