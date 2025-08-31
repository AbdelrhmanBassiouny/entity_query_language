# EQL for SQL Experts: A Practical Guide

If you are fluent in SQL, this guide will help you become productive with the Entity Query Language (EQL) quickly.
EQL provides a relational, pythonic way to express queries and rule-based reasoning directly in Python, while
minimizing boilerplate. A key difference from SQL is that joins are implicit: relationships are expressed as
conditions over Python objects, and EQL finds matching assignments for you.

This guide maps familiar SQL ideas (SELECT, WHERE, JOIN, EXISTS/IN, boolean logic, uniqueness) to EQL constructs.


## TL;DR: SQL → EQL mental model

- FROM aliasing → let(name, type_, domain): declare placeholders (think table aliases) and their data source.
- SELECT one column/entity → entity(selected_var, ...conditions): choose which variable to return.
- SELECT multiple columns → set_of((var1, var2, ...), ...conditions): return a tuple of variables.
- WHERE predicates → and_(...), or_(...), not_(...), and Python methods (e.g., startswith), plus contains/in_.
- JOIN (INNER) → express relationships with equality between placeholders/attributes; EQL joins implicitly.
- IN / EXISTS → in_(x, collection) or contains(collection, x); presence is implicit by using a placeholder.
- Enforce exactly one row → the(...).evaluate(): raises if multiple matches; an(...) allows many.

References to examples in this documentation set:
- Example with the: uniqueness, like asserting exactly one result.
- Example with And + Or: complex boolean logic.
- Example with Not: negation.
- Example with Joining Multiple Sources: implicit joins via relationships.
- Example with Inference: derive structured objects from matched patterns.


## Minimal example: SELECT … FROM … WHERE …

SQL

~~~~sql
SELECT body.*
FROM bodies AS body
WHERE body.name = 'Body2';
~~~~

EQL

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

# FROM bodies AS body
body = let(name="body", type_=Body, domain=world.bodies)

# SELECT body WHERE body.name == "Body2"
results_generator = an(entity(body, body.name == "Body2")).evaluate()
results = list(results_generator)
assert results[0].name == "Body2"
```

Notes
- let declares a placeholder (variable) and its domain (data source) — analogous to FROM with an alias.
- entity selects which variable (or tuple of variables) to return.
- an(...) returns all matches; evaluate() yields a generator of results.


## LIKE and string predicates

SQL

~~~~sql
SELECT body.*
FROM bodies AS body
WHERE body.name LIKE 'Body%'
  AND body.name LIKE '%2';
~~~~

EQL

```python
from entity_query_language import entity, an, let, and_
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

body = let("body", Body, domain=world.bodies)
results_generator = an(entity(body,
                              and_(body.name.startswith("Body"),
                                   body.name.endswith("2"))
                              )).evaluate()
results = list(results_generator)
```

Tip: You can also use contains(x, y) or in_(x, collection) for containment tests.


## IN and EXISTS

SQL (IN)

~~~~sql
SELECT body.*
FROM bodies AS body
WHERE body.name IN ('Container1', 'Handle1');
~~~~

EQL

```python
from entity_query_language import entity, an, let, in_, contains
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

body = let("body", Body, domain=world.bodies)
names = {"Container1", "Handle1"}
in_results_generator = an(entity(body, in_(body.name, names))).evaluate()
contains_results_generator = an(entity(body, contains(names, body.name))).evaluate()
```

EXISTS in SQL is naturally expressed by just introducing a placeholder and relating it in conditions. If the
relationships can be satisfied, it “exists.”


## Implicit JOINs via relationships

In SQL you join tables explicitly. In EQL you state the relationships between placeholders and attributes; the join is
inferred.

SQL (conceptually)

~~~~sql
SELECT parent_container.*, prismatic_connection.*, drawer_body.*, fixed_connection.*, handle.*
FROM bodies AS parent_container
JOIN prismatic AS prismatic_connection ON parent_container.id = prismatic_connection.parent_id
JOIN bodies AS drawer_body ON drawer_body.id = prismatic_connection.child_id
JOIN fixed AS fixed_connection ON drawer_body.id = fixed_connection.parent_id
JOIN bodies AS handle ON handle.id = fixed_connection.child_id;
~~~~

EQL

```python
from entity_query_language import entity, an, let, and_, set_of
from dataclasses import dataclass, field
from typing_extensions import List


@dataclass
class Body:
    name: str


@dataclass
class Connection:
    parent: Body
    child: Body


@dataclass
class Prismatic(Connection):
    ...


@dataclass
class Fixed(Connection):
    ...


@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


# Construct a small world
world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
c1_c2 = Prismatic(world.bodies[0], world.bodies[1])
c2_h2 = Fixed(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

# Declare placeholders (FROM with aliases)
parent_container = let("parent_container", Body, domain=world.bodies)
prismatic_connection = let("prismatic_connection", Prismatic, domain=world.connections)
drawer_body = let("drawer_body", Body, domain=world.bodies)
fixed_connection = let("fixed_connection", Fixed, domain=world.connections)
handle = let("handle", Body, domain=world.bodies)

# SELECT (parent_container, prismatic_connection, drawer_body, fixed_connection, handle) WHERE relationships hold
results_generator = an(set_of((parent_container, prismatic_connection, drawer_body, fixed_connection, handle),
                              and_(parent_container == prismatic_connection.parent,
                                   drawer_body == prismatic_connection.child,
                                   drawer_body == fixed_connection.parent,
                                   handle == fixed_connection.child)
                              )).evaluate()
results = list(results_generator)
```

Notice how equality constraints play the role of JOIN conditions. The set_of(...) returns a tuple-like result where you
can access each variable by its identity if needed.


## Selecting multiple outputs (projection of multiple columns)

- Use entity(x, ...) to return one variable.
- Use set_of((x, y, ...), ...) to return several variables together (akin to SELECT col1, col2, ...).


## Enforcing uniqueness (like expecting exactly one row)

- an(...) returns all matching assignments.
- the(...) asserts there is exactly one satisfying assignment; it raises if there are 0 or >1.

```python
from entity_query_language import entity, the, let
from entity_query_language.failures import MultipleSolutionFound
from dataclasses import dataclass
from typing_extensions import List

@dataclass(unsafe_hash=True)
class Body:
    name: str

@dataclass(eq=False)
class World:
    id_: int
    bodies: List[Body]

# Two bodies with the same name to demonstrate uniqueness enforcement
world = World(1, [Body("Body1"), Body("Body1")])

body = let("body", Body, domain=world.bodies)
# Expect exactly one
try:
    only_body = the(entity(body, body.name == "Body1")).evaluate()
    assert False  # should not reach here due to MultipleSolutionFound
except MultipleSolutionFound:
    pass
```

This is analogous to a query where you expect a single-row result (e.g., by primary key). Unlike LIMIT 1,
`the(...)` enforces uniqueness by raising on multiple solutions.


## Rule-based inference (beyond SQL)

EQL can construct new objects from matched patterns using symbolic_mode and @symbol. Conceptually, this is more like
rule-based reasoning than SQL aggregation. See the Inference example for full context.

```python
from entity_query_language import entity, an, let, and_, symbolic_mode, symbol
from dataclasses import dataclass, field
from typing_extensions import List


@dataclass
class Body:
    name: str


@dataclass
class Connection:
    parent: Body
    child: Body


@dataclass
class Prismatic(Connection):
    ...


@dataclass
class Fixed(Connection):
    ...


@dataclass
class World:
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


# Build a small world
world = World(1, [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")])
c1_c2 = Prismatic(world.bodies[0], world.bodies[1])
c2_h2 = Fixed(world.bodies[1], world.bodies[3])
world.connections = [c1_c2, c2_h2]

# Placeholders
parent_container = let("parent_container", Body, domain=world.bodies)
prismatic_connection = let("prismatic_connection", Prismatic, domain=world.connections)
drawer_body = let("drawer_body", Body, domain=world.bodies)
fixed_connection = let("fixed_connection", Fixed, domain=world.connections)
handle = let("handle", Body, domain=world.bodies)


@symbol
@dataclass
class Drawer:
    handle: Body
    body: Body


with symbolic_mode():
    results_generator = an(entity(Drawer(handle=handle, body=drawer_body),
                                  and_(parent_container == prismatic_connection.parent,
                                       drawer_body == prismatic_connection.child,
                                       drawer_body == fixed_connection.parent, handle == fixed_connection.child)
                                  )).evaluate()
results = list(results_generator)
assert results and results[0].body.name == "Container2" and results[0].handle.name == "Handle2"
```


## Tips for SQL users

- Think: placeholders (let) are your table aliases, and their domain is your FROM source (Python collections).
- Build predicates using and_/or_/not_, equality between attributes, and common Python string/collection methods.
- Default “join” is whatever satisfies your equality constraints — you don’t spell JOIN explicitly.
- Materialize results with list(...). The evaluator yields a generator so you can stream or iterate.
- For returning multiple columns, use set_of with a tuple of placeholders.
- For exact-one expectations, use the(...); otherwise use an(...).

For more, see the example pages in this documentation:
- Example with the
- Example with And + Or
- Example with Not
- Example with Joining Multiple Sources
- Example with Inference
