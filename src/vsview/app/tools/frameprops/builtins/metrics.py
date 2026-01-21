from __future__ import annotations

from ..categories import CategoryMatcher
from ..formatters import FormatterProperty

__all__ = ["METRICS_CATEGORY", "METRICS_FORMATTERS"]


METRICS_CATEGORY = CategoryMatcher(
    name="Metrics",
    priority=100,
    order=20,
    exact_matches={
        # Scene detection
        "_SceneChangeNext",
        "_SceneChangePrev",
    },
    prefixes={
        # PlaneStats
        "PlaneStats",
    },
)


METRICS_FORMATTERS = [
    # Scene change detection
    FormatterProperty(
        prop_key="_SceneChangeNext",
        value_formatter={0: "Current Scene", 1: "End of Scene"},
    ),
    FormatterProperty(
        prop_key="_SceneChangePrev",
        value_formatter={0: "Current Scene", 1: "Start of Scene"},
    ),
]
