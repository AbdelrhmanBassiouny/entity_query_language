import pytest

from entity_query_language.predicate import Predicate


def test_inverse_of_sets_back_reference():
    class ParentOf(Predicate):
        def __call__(self):
            return True

    class ChildOf(Predicate):
        inverse_of = ParentOf

        def __call__(self):
            return True

    assert ParentOf.inverse_of is ChildOf
    assert ChildOf.inverse_of is ParentOf


def test_inverse_of_type_validation():
    # inverse_of must be a Predicate subclass
    with pytest.raises(TypeError):
        class InvalidInverse(Predicate):
            inverse_of = object  # not a Predicate subclass

            def __call__(self):
                return True
