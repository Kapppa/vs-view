from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from io import TextIOWrapper
from typing import TYPE_CHECKING, BinaryIO, ClassVar, NamedTuple

from .specs import hookimpl

if TYPE_CHECKING:
    from PySide6.QtGui import QColor

    from .models import SceneRow, UnifiedRange

__all__ = ["FileFilter", "Parser", "SceneRow", "UnifiedRange", "borrowed_text_wrapper", "hookimpl"]


class FileFilter(NamedTuple):
    """Named tuple representing a file filter for dialogs."""

    label: str
    """The display label for the filter."""
    suffix: str | Sequence[str]
    """The file extension suffix."""


_FileFilter = FileFilter


@contextmanager
def borrowed_text_wrapper(
    stream: BinaryIO,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Iterator[TextIOWrapper]:
    wrapper = TextIOWrapper(stream, encoding=encoding, errors=errors)
    try:
        yield wrapper
    finally:
        wrapper.detach()


class Parser(ABC):
    FileFilter: type[_FileFilter] = _FileFilter
    filter: ClassVar[_FileFilter]

    @abstractmethod
    def parse(self, io: BinaryIO, name: str, fps: Fraction) -> SceneRow | Sequence[SceneRow]: ...

    if TYPE_CHECKING:

        def get_color(self) -> QColor: ...

