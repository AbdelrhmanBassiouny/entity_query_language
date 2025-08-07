# Entity Query Language


Welcome to the Entity Query Language package!

EQL is a relational query language that is pythonic, and intuitive. This serves as a front end
to other query languages like sql or prolog.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.


## Example Usage

An important feature of EQL is that you do not need to do operations like JOIN in SQL, this is performed implicitly.
EQL tries to mirror your intent in a query statement with as less boiler plate code as possiple.
For example an attribute access with and equal check to another value is just how you expect:

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

results = an(entity(body:=Body, domain=world.bodies), body.name == "Body2")
assert results[0][body].name == "Body2"
```

where this creates a body variable that gets its values from world.bodies, and filters them to have their att "name"
equal to "Body1".ample shows generic usage of the Ripple Down Rules to classify objects in a propositional setting.

## To Cite:

```bib
@software{bassiouny2025eql,
author = {Bassiouny, Abdelrhman},
title = {Entity-Query-Language},
url = {https://github.com/AbdelrhmanBassiouny/entity_query_language},
version = {0.0.1},
}
```