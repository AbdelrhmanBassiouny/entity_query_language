# Entity Query Language


Welcome to the Entity Query Language package!

EQL is a relational query language that is pythonic, and intuitive.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.


## Installation

```bash
pip install entity_query_language
```
If you want to use the visualization feature, you will also need to install [rustworkx_utils](https://github.com/AbdelrhmanBassiouny/rustworkx_utils).
```bash
pip install rustworkx_utils
```

# Example Usage

## Basic Example
An important feature of EQL is that you do not need to do operations like JOIN in SQL, this is performed implicitly.
EQL tries to mirror your intent in a query statement with as less boiler plate code as possiple.
For example an attribute access with and equal check to another value is just how you expect:

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


world = World(1, [Body("Body1"), Body("Body2")])

with symbolic_mode():
    body = let(type_=Body, domain=world.bodies)
    query = an(entity(body, contains(body.name, "2"),
                      body.name.startswith("Body"))
               )
results = list(query.evaluate())
assert len(results) == 1
assert results[0].name == "Body2"
```

where this creates a body variable that gets its values from world.bodies, and filters them to have their att "name"
equal to "Body1".ample shows generic usage of the Ripple Down Rules to classify objects in a propositional setting.

Notice that it is possible to use both provided helper methods by EQL and other methods that are accessible through your
object instance and use them as part of the query conditions.

## More Example Usage

- [Example with `the`](example_with_the.md): This example shows how to use `the` entity wrapper instead of `an`.
- [Example with `And` + `OR`](example_with_and_or.md): This shows an example of using `And` with `Or` together.
- [Example with `Not`](example_with_not.md): This shows an example of using `Not`.
- [Example with Joining Multiple Sources](example_with_joining_multiple_sources.md): This shows an example of using and joining multiple sources in your query.
- [Example with Nested Queries](example_with_nested_queries.md): This shows how to compose queries by nesting queries inside others.
- [Example with Rule Inference](example_with_rule_inference.md): This shows an example of writing inference rules in EQL.
- [Example with Rule Tree](example_with_rule_tree.md): This shows how to build and visualize a rule tree.
- [Example with Predicates](example_with_predicate.md): This shows how to write and use reusable predicates in queries.
- [Example with Predicate Style Query/Rule](example_with_predicate_style_query.md): This shows queries and rules written in predicate (functional) style.
- [Example with Cached Symbols](example_with_cached_symbols.md): This shows using cached @symbol instances without providing a domain.
- [Example with Indexing](example_with_indexing.md): This shows capturing __getitem__ (indexing) in symbolic expressions.
- [Example with Flatten](example_with_flatten.md): This shows the usage of the flatten() operation on nested variables.
- [Example with Concatenate](example_with_concatenate.md): This shows the concatenate operation that combines inner
  iterables into one.

## To Cite:

```bib
@software{bassiouny2025eql,
author = {Bassiouny, Abdelrhman},
title = {Entity-Query-Language},
url = {https://github.com/AbdelrhmanBassiouny/entity_query_language},
version = {3.1.0},
}
```