from __future__ import annotations

from dataclasses import dataclass
from typing import List

from entity_query_language import symbolic_mode, symbol, in_, a
from entity_query_language.property_descriptor import PropertyDescriptor, Thing
from entity_query_language.symbolic import From


@dataclass(frozen=True)
class MemberOf(PropertyDescriptor):
    ...


@dataclass(frozen=True)
class WorksFor(MemberOf):
    ...


@dataclass(unsafe_hash=True)
class Organization(Thing):
    name: str


@dataclass(unsafe_hash=True)
class Company(Organization):
    ...


@dataclass(eq=False)
class Person(Thing):
    name: str

@dataclass(eq=False)
class Employee(Person):
    works_for: List[Organization] = WorksFor(default_factory=list)


def test_query_on_descriptor_field_filters():
    org1 = Organization("ACME")
    org2 = Company("ABC")

    people = [
        Person("John"),
        Person("Jane"),
    ]
    people[0].works_for = [org1]
    people[1].works_for = [org2]

    with symbolic_mode():
        query = a(person := Person(From(people)), in_(Organization("ACME"), person.works_for))
    results = list(query.evaluate())
    assert [p.name for p in results] == ["John"]


def test_query_on_descriptor_inheritance():
    org1 = Organization("ACME")
    org2 = Company("ABC")

    people = [
        Employee("John"),
        Employee("Jane"),
    ]
    people[0].works_for = [org1]
    people[1].works_for = [org2]

    with symbolic_mode():
        query = a(person := Person(From(people)), MemberOf(person, Organization("ACME")))
    results = list(query.evaluate())
    assert [p.name for p in results] == ["John"]
