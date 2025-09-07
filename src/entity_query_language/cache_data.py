from __future__ import annotations

from copy import copy

from . import logger

"""
Cache utilities.

This module provides caching datastructures and utilities. In addision it 
provides simple counters and timers used to profile the internal
caching layer. It also exposes a runtime switch to enable/disable caching.
"""
import contextvars
from collections import defaultdict, UserDict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterable


@dataclass
class CacheCount:
    """
    Counter for named cache-related events.

    :ivar values: Mapping from counter name to its integer value.
    :vartype values: Dict[str, int]
    """
    values: Dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def update(self, name: str) -> None:
        """
        Increment a named counter.

        :param name: Counter name to increment.
        :type name: str
        :return: None
        :rtype: None
        """
        self.values[name] += 1


@dataclass
class CacheTime:
    """
    Aggregator for named timing values (in seconds).

    :ivar values: Mapping from timer name to accumulated seconds.
    :vartype values: Dict[str, float]
    """
    values: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.0))

    def add(self, name: str, seconds: float) -> None:
        """
        Add elapsed time to a named timer.

        :param name: Timer name.
        :type name: str
        :param seconds: Elapsed time to add in seconds.
        :type seconds: float
        :return: None
        :rtype: None
        """
        self.values[name] += seconds


cache_enter_count = CacheCount()
cache_search_count = CacheCount()
cache_match_count = CacheCount()
cache_lookup_time = CacheTime()
cache_update_time = CacheTime()

# Runtime switch to enable/disable caching paths
_caching_enabled = contextvars.ContextVar("caching_enabled", default=True)

def enable_caching() -> None:
    """
    Enable the caching fast-paths for query evaluation.

    :return: None
    :rtype: None
    """
    _caching_enabled.set(True)


def disable_caching() -> None:
    """
    Disable the caching fast-paths for query evaluation.

    :return: None
    :rtype: None
    """
    _caching_enabled.set(False)


def is_caching_enabled() -> bool:
    """
    Check whether caching is currently enabled.

    :return: True if caching is enabled; False otherwise.
    :rtype: bool
    """
    return _caching_enabled.get()


def cache_profile_report() -> Dict[str, Dict[str, float]]:
    """
    Produce a snapshot of current cache statistics.

    :return: Dictionary with counters and timers grouped by kind.
    :rtype: Dict[str, Dict[str, float]]
    """
    return {
        "enter_count": dict(cache_enter_count.values),
        "search_count": dict(cache_search_count.values),
        "match_count": dict(cache_match_count.values),
        "lookup_time_seconds": dict(cache_lookup_time.values),
        "update_time_seconds": dict(cache_update_time.values),
    }


@dataclass(eq=False)
class ALL:
    """
    Sentinel that compares equal to any other value.

    This is used to signal wildcard matches in hashing/containment logic.
    """
    def __eq__(self, other):
        """Always return True."""
        return True

    def __hash__(self):
        """Hash based on object identity to remain unique as a sentinel."""
        return hash(id(self))


All = ALL()


@dataclass
class SeenSet:
    """
    Tracks sets of partial assignments to avoid duplicate processing.

    Each assignment is a dict of key->value pairs. Missing keys act as wildcards.

    :ivar seen: Collected assignment constraints.
    :ivar all_seen: Becomes True when an empty assignment is added, meaning any
                    assignment is considered seen.
    """
    seen: List[Any] = field(default_factory=list, init=False)
    all_seen: bool = field(default=False, init=False)

    def add(self, assignment):
        """
        Add an assignment (dict of key→value).
        Missing keys are implicitly wildcards.
        Example: {"k1": "v1"} means all k2,... are allowed
        """
        if not self.all_seen:
            self.seen.append(assignment)
            if not assignment:
                self.all_seen = True

    def check(self, assignment):
        """
        Check if an assignment (dict) is covered by seen entries.
        """
        if self.all_seen:
            return True
        for constraint in self.seen:
            if all(assignment[k] == v if k in assignment else False for k, v in constraint.items()):
                return True
        return False

    def clear(self):
        self.seen.clear()
        self.all_seen = False


class CacheDict(UserDict):
    ...


@dataclass
class IndexedCache:
    """
    A hierarchical cache keyed by a fixed sequence of indices.

    It supports insertion of outputs under partial assignments and retrieval with
    wildcard handling using the ALL sentinel.

    :ivar keys: Ordered list of integer keys to index the cache.
    :ivar seen_set: Helper to track assignments already checked.
    :ivar cache: Nested mapping structure storing cached results.
    :ivar enter_count: Diagnostic counter for retrieval entries.
    :ivar search_count: Diagnostic counter for wildcard searches.
    """
    keys: List[int] = field(default_factory=list)
    seen_set: SeenSet = field(default_factory=SeenSet, init=False)
    cache: CacheDict = field(default_factory=CacheDict, init=False)
    enter_count: int = field(default=0, init=False)
    search_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Ensure keys remain sorted for deterministic traversal."""
        self.keys.sort()

    def insert(self, assignment: Dict, output: Any) -> None:
        """
        Insert an output under the given partial assignment.

        Missing keys are filled with the ALL sentinel.

        :param assignment: Mapping from key index to concrete value.
        :type assignment: Dict
        :param output: Cached value to store at the leaf.
        :type output: Any
        :return: None
        :rtype: None
        """
        assignment = dict(assignment)
        cache = self.cache
        for k_idx, k in enumerate(self.keys):
            if k not in assignment.keys():
                assignment[k] = All
                logger.debug(f"Missing key {k} in assignment {assignment}, using {All}")
            if k_idx < len(self.keys) - 1:
                if (k, assignment[k]) not in cache:
                    cache[(k, assignment[k])] = CacheDict()
                cache = cache[(k, assignment[k])]
            else:
                cache[(k, assignment[k])] = output

    def check(self, assignment: Dict) -> bool:
        """
        Check if seen entries cover an assignment (dict).

        :param assignment: The assignment to check.
        """
        assignment = {k: v for k, v in assignment.items() if k in self.keys}
        seen = self.seen_set.check(assignment)
        if not seen:
            self.seen_set.add(assignment)
        return seen

    def retrieve(self, assignment, cache=None, key_idx=0, result: Dict = None) -> Iterable:
        """
        Retrieve leaf results matching a (possibly partial) assignment.

        This yields tuples of (resolved_assignment, value) when a leaf is found.

        :param assignment: Partial mapping from key index to values.
        :param cache: Internal recursion parameter; the current cache node.
        :param key_idx: Internal recursion parameter; current key index position.
        :param result: Internal accumulator for building a full assignment.
        :return: Generator of (assignment, value) pairs.
        :rtype: Iterable
        """
        result = result or copy(assignment)
        if cache is None:
            cache = self.cache
            self.enter_count += 1
        if isinstance(cache, CacheDict) and len(cache) == 0:
            return
        key = self.keys[key_idx]
        while key in assignment:
            try:
                cache = cache[(key, assignment[key])]
            except KeyError:
                for cache_key, cache_val in cache.items():
                    if isinstance(cache_key[1], ALL):
                        yield from self._yield_result(assignment, cache_val, key_idx, result)
                    else:
                        self.search_count += 1
                return
            if key_idx+1 < len(self.keys):
                key_idx = key_idx + 1
                key = self.keys[key_idx]
            else:
                break
        if key not in assignment:
            for cache_key, cache_val in cache.items():
                if not isinstance(cache_key[1], ALL):
                    result = copy(result)
                    result[key] = cache_key[1]
                yield from self._yield_result(assignment, cache_val, key_idx, result)
        else:
            yield result, cache

    def clear(self):
        self.cache.clear()
        self.seen_set.clear()

    def _yield_result(self, assignment: Dict, cache_val: Any, key_idx: int, result: Dict[int, Any]):
        """
        Internal helper to descend into cache and yield concrete results.

        :param assignment: Original partial assignment.
        :param cache_val: Current cache node or value.
        :param key_idx: Current key index.
        :param result: Accumulated assignment.
        :return: Yields (assignment, value) when reaching leaves.
        """
        if isinstance(cache_val, CacheDict):
            self.search_count += 1
            yield from self.retrieve(assignment, cache_val, key_idx + 1, result)
        else:
            yield result, cache_val
