from __future__ import annotations

from ..categories import CategoryMatcher
from ..formatters import FormatterProperty

FIELD_CATEGORY = CategoryMatcher(
    name="Field",
    priority=50,
    order=30,
    exact_matches={
        "_Combed",
        "_Field",
    },
)


FIELD_FORMATTERS = [
    FormatterProperty(
        prop_key="_Combed",
        value_formatter={0: "No", 1: "Yes"},
    ),
    FormatterProperty(
        prop_key="_Field",
        value_formatter={0: "Bottom Field", 1: "Top Field"},
    ),
]
