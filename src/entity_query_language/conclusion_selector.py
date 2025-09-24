from __future__ import annotations

import typing
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Iterable

from .cache_data import SeenSet, is_caching_enabled
from .conclusion import Conclusion
from .hashed_data import HashedIterable, HashedValue
from .rxnode import ColorLegend
from .symbolic import LogicalOperator, SymbolicExpression, ElseIf, Union as EQLUnion, Literal


@dataclass(eq=False)
class ConclusionSelector(LogicalOperator, ABC):
    """
    Base class for logical operators that may carry and select conclusions.

    Tracks whether certain conclusion-combinations were already produced so
    they are not duplicated across truth branches.
    """
    concluded_before: Dict[bool, SeenSet] = field(default_factory=lambda: {True: SeenSet(), False: SeenSet()},
                                                  init=False)

    def update_conclusion(self, output: Dict[int, HashedValue], conclusions: typing.Set[Conclusion]) -> None:
        if not conclusions:
            return
        required_vars = HashedIterable()
        for conclusion in conclusions:
            vars_ = conclusion._unique_variables_.filter(lambda v: not isinstance(v.value, Literal))
            required_vars.update(vars_)
        required_output = {k: v for k, v in output.items() if k in required_vars}
        if not self.concluded_before[not self._is_false_].check(required_output):
            self._conclusion_.update(conclusions)
            self.concluded_before[not self._is_false_].add(required_output)

    def _copy_expression_(self, postfix: str) -> SymbolicExpression:
        cp = super()._copy_expression_(postfix)
        cp.concluded_before = {True: SeenSet(), False: SeenSet()}
        return cp

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("ConclusionSelector", '#eded18')


@dataclass(eq=False)
class ExceptIf(ConclusionSelector):
    """
    Conditional branch that yields left unless the right side produces values.

    This encodes an "except if" behavior: when the right condition matches,
    the left branch's conclusions/outputs are excluded; otherwise, left flows through.
    """

    @lru_cache(maxsize=None)
    def _required_variables_from_child_(self, child: Optional[SymbolicExpression] = None, when_true: bool = True):
        if not child:
            child = self.left
        required_vars = HashedIterable()
        when_false = not when_true
        if child is self.left:
            if when_true:
                required_vars.update(self.right._unique_variables_)
            for conc in self.left._conclusion_.union(self.right._conclusion_):
                required_vars.update(conc._unique_variables_)
        elif child is self.right:
            if when_true:
                for conc in self.right._conclusion_:
                    required_vars.update(conc._unique_variables_)
            if when_false and not self.left._is_false_:
                for conc in self.left._conclusion_:
                    required_vars.update(conc._unique_variables_)
        required_vars.update(self._parent_._required_variables_from_child_(self, when_true))
        for conc in self._conclusion_:
            required_vars.update(conc._unique_variables_)
        return required_vars

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None, yield_when_false: bool = False) -> Iterable[Dict[int, HashedValue]]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """

        # init an empty source if none is provided
        sources = sources or HashedIterable()
        self._yield_when_false_ = yield_when_false

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources, yield_when_false=self._yield_when_false_)
        for left_value in left_values:

            left_value.update(sources)

            self._is_false_ = self.left._is_false_
            if self._is_false_:
                if self._yield_when_false_:
                    if not self._is_duplicate_output_(left_value):
                        yield left_value
                continue

            if is_caching_enabled() and self.right_cache.check(left_value):
                yield from self.yield_final_output_from_cache(left_value, self.right_cache)
                continue

            right_yielded = False
            for right_value in self.right._evaluate__(left_value, yield_when_false=False):
                right_yielded = True
                self._conclusion_.update(self.right._conclusion_)
                output = left_value.copy()
                output.update(right_value)
                yield output
                self._conclusion_.clear()
            if not right_yielded:
                self._conclusion_.update(self.left._conclusion_)
                yield left_value
                self._conclusion_.clear()


@dataclass(eq=False)
class Alternative(ElseIf, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch
    is selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.
    """

    def _is_duplicate_output_(self, output: Dict[int, HashedValue]) -> bool:
        # For alternatives, avoid suppressing outputs based solely on variable values,
        # as different branches may yield different conclusions for the same bindings.
        # Let ConclusionSelector.update_conclusion handle deduplication of conclusions.
        return False

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None, yield_when_false: bool = False) -> Iterable[Dict[int, HashedValue]]:
        outputs = super()._evaluate__(sources, yield_when_false=yield_when_false)
        for output in outputs:
            left_is_true = not self.left._is_false_
            right_is_true = not self.right._is_false_
            if left_is_true:
                self.update_conclusion(output, self.left._conclusion_)
            elif right_is_true:
                self.update_conclusion(output, self.right._conclusion_)
            yield output
            self._conclusion_.clear()


@dataclass(eq=False)
class Next(EQLUnion, ConclusionSelector):
    """
    A Union conclusion selector that always evaluates the left and right branches and combines their results.
    """

    def _evaluate__(self, sources: Optional[Dict[int, HashedValue]] = None, yield_when_false: bool = False) -> Iterable[Dict[int, HashedValue]]:
        outputs = super()._evaluate__(sources, yield_when_false=yield_when_false)
        for output in outputs:
            if self.left_evaluated:
                self.update_conclusion(output, self.left._conclusion_)
            if self.right_evaluated:
                self.update_conclusion(output, self.right._conclusion_)
            yield output
            self._conclusion_.clear()