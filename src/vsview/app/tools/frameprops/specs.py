from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from .categories import CategoryMatcher, IterCategoryMatcher
    from .formatters import FormatterProperty, IterFormatter

hookspec = pluggy.HookspecMarker("vsview.frameprops")
hookimpl = pluggy.HookimplMarker("vsview.frameprops")


@hookspec
def vsview_frameprops_register_category_matchers() -> CategoryMatcher | IterCategoryMatcher:
    raise NotImplementedError


@hookspec
def vsview_frameprops_register_formatter_properties() -> FormatterProperty | IterFormatter:
    raise NotImplementedError
