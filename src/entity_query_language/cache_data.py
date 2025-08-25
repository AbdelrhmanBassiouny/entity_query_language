from __future__ import annotations

import contextvars
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CacheCount:
    values: Dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def update(self, name: str):
        self.values[name] += 1


@dataclass
class CacheTime:
    values: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.0))

    def add(self, name: str, seconds: float):
        self.values[name] += seconds


_cache_enter_count = CacheCount()
_cache_search_count = CacheCount()
_cache_match_count = CacheCount()
_cache_lookup_time = CacheTime()
_cache_update_time = CacheTime()

# Runtime switch to enable/disable caching paths
_caching_enabled = contextvars.ContextVar("caching_enabled", default=True)

def enable_caching():
    _caching_enabled.set(True)


def disable_caching():
    _caching_enabled.set(False)


def is_caching_enabled() -> bool:
    return _caching_enabled.get()


def cache_profile_report():
    return {
        "enter_count": dict(_cache_enter_count.values),
        "search_count": dict(_cache_search_count.values),
        "match_count": dict(_cache_match_count.values),
        "lookup_time_seconds": dict(_cache_lookup_time.values),
        "update_time_seconds": dict(_cache_update_time.values),
    }
