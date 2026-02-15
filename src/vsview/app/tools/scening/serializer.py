from collections.abc import Iterable
from typing import BinaryIO

from .api import Serializer, borrowed_text_wrapper
from .models import UnifiedRange


class OGMSerializer(Serializer):
    filter = Serializer.FileFilter("OGM Chapters", "txt")

    def serialize(self, io: BinaryIO, ranges: Iterable[UnifiedRange]) -> None:
        with borrowed_text_wrapper(io) as wrapper:
            for i, r in enumerate(ranges, 1):
                timestamp = r.as_times()[0].to_ts()
                label = r.label or f"Chapter {i:02}"

                wrapper.write(f"CHAPTER{i:02}={timestamp}\n")
                wrapper.write(f"CHAPTER{i:02}NAME={label}\n")


internal_serializers: list[Serializer] = [
    OGMSerializer(),
]
