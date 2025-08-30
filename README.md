# Entity Query Language (EQL)

EQL is a relational query language that is pythonic, and intuitive.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.

## Installation

```bash
pip install entity_query_language
```
## Documentation

Read the documentation [here](https://abdelrhmanbassiouny.github.io/entity_query_language/).

## Example Usage

An important feature of EQL is that you do not need to do operations like JOIN in SQL, this is performed implicitly.
EQL tries to mirror your intent in a query statement with as less boiler plate code as possiple.
For example an attribute access with an equality check to another value is as simple as using python's dot notation with
the equality operator. For example:

```python
from entity_query_language import entity, an, let
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

results_generator = an(entity(body := let("body", type_=Body, domain=world.bodies), body.name == "Body2")).evaluate()
results = list(results_generator)
assert results[0].name == "Body2"
```

where this creates a body variable that gets its values from world.bodies, and filters them to have their att "name"
equal to "Body1".

## To Cite:

```bib
@software{bassiouny2025eql,
author = {Bassiouny, Abdelrhman},
title = {Entity-Query-Language},
url = {https://github.com/AbdelrhmanBassiouny/entity_query_language},
version = {1.0.0},
}
```
