from __future__ import annotations

from dataclasses import dataclass
from typing import List

from entity_query_language.property_descriptor import OntologyMeta, PropertyDescriptor
from entity_query_language.symbolic import From
from entity_query_language import an, entity, let, symbolic_mode, symbol, in_, a


@dataclass(frozen=True)
class MemberOf(PropertyDescriptor):
    ...

@dataclass(frozen=True)
class WorksFor(MemberOf):
    ...


@dataclass(unsafe_hash=True)
class Organization:
    name: str


@dataclass(unsafe_hash=True)
class Company(Organization):
    ...


@symbol
@dataclass(unsafe_hash=True)
class Person(metaclass=OntologyMeta):
    name: str
    worksFor: List[Organization] = WorksFor()


def test_query_on_descriptor_field_filters_correctly():
    org1 = Organization("ACME")
    org2 = Company("ABC")

    people = [
        Person("John"),
        Person("Jane"),
    ]
    people[0].worksFor = [org1]
    people[1].worksFor = [org2]

    with symbolic_mode():
        query = a(person := Person(From(people)), in_(Organization("ACME"), person.worksFor))
    results = list(query.evaluate())
    assert [p.name for p in results] == ["John"]


def test_query_on_descriptor_field_filters_correctly():
    org1 = Organization("ACME")
    org2 = Company("ABC")

    people = [
        Person("John"),
        Person("Jane"),
    ]
    people[0].worksFor = [org1]
    people[1].worksFor = [org2]

    with symbolic_mode():
        query = a(person := Person(From(people)), MemberOf(person, Organization("ACME")))
    results = list(query.evaluate())
    assert [p.name for p in results] == ["John"]
