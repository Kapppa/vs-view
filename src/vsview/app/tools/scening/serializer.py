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


class PythonListFramesSerializer(Serializer):
    filter = Serializer.FileFilter("Python List (Frames)", "txt")

    def serialize(self, io: BinaryIO, ranges: Iterable[UnifiedRange]) -> None:
        with borrowed_text_wrapper(io) as wrapper:
            wrapper.write(str([r.as_frames() for r in ranges]))


class PythonListTimestampsSerializer(Serializer):
    filter = Serializer.FileFilter("Python List (Timestamps)", "txt")

    def serialize(self, io: BinaryIO, ranges: Iterable[UnifiedRange]) -> None:
        with borrowed_text_wrapper(io) as wrapper:
            wrapper.write(
                str([tuple(ts.to_ts("{H:02d}:{M:02d}:{S:02d}.{ms:06d}") for ts in r.as_times()) for r in ranges])
            )


internal_serializers: list[Serializer] = [
    OGMSerializer(),
    PythonListFramesSerializer(),
    PythonListTimestampsSerializer(),
]
