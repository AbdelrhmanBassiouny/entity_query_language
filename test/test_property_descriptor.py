from __future__ import annotations

from dataclasses import dataclass
from typing import List

from entity_query_language.property_descriptor import PropertyDescriptor, Thing


# Concrete descriptor used in tests
@dataclass(frozen=True)
class WorksFor(PropertyDescriptor):
    ...


@dataclass
class Organization(Thing):
    name: str


@dataclass
class Company(Organization):
    ...


@dataclass
class Person(Thing):
    name: str
    worksForOrg: List[Organization] = WorksFor()
    worksForCompany: List[Company] = WorksFor()


@dataclass
class Employee(Person):
    worksForOrg: List[Organization] = WorksFor(default_factory=lambda: [Company("Unknown")])


def test_descriptor_stores_per_instance_values_and_metadata():
    person = Person("John")
    organization = Organization("ACME")
    company = Company("ABC Corp")

    # Per instance storage
    person.worksForOrg = [organization]
    person2 = Person("Jane")
    person2.worksForCompany = [company]

    assert person.worksForOrg == [organization]

    # Accessing worksForOrg on a different person should raise until explicitly set
    try:
        _ = person2.worksForOrg
        assert False, "Should not be able to access worksForOrg on person2 before it is set"
    except AttributeError:
        pass

    # default factory on subclass
    employee = Employee("Ahmed")
    assert employee.worksForOrg == [Company("Unknown")]

    # setattr should work
    setattr(person2, "worksForCompany", [Company("SetAttr")])
    assert person2.worksForCompany == [Company("SetAttr")]

    # Class access returns the descriptor
    assert isinstance(person2.__class__.worksForOrg, WorksFor)

    # Domain types and range types
    assert Person in Person.worksForOrg.domain_types
    assert Organization in person.__class__.worksForOrg.range_types
    # WorksFor on Person and Employee share class variables
    assert person2.__class__.worksForCompany.domain_types == WorksFor.domain_types


def test_nullable_and_name_attributes():
    # The range types contain no None by default
    wf = Person.worksForOrg
    assert wf.nullable is False
    assert wf.name == "worksFor"
