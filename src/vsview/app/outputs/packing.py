"""RGB packing implementations for VapourSynth to Qt conversion."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum
from functools import cache
from logging import DEBUG, getLogger
from typing import Any, ClassVar, Literal

import vapoursynth as vs
from PySide6.QtGui import QImage
from vspackrgb.helpers import get_plane_buffer, packrgb

from ...vsenv import create_environment
from ..settings import SettingsManager

logger = getLogger(__name__)


class AlphaNotImplementedError(NotImplementedError):
    """Alpha packing hasn't been implemented for this packer"""

    packer: Packer

    def __init__(self, packer: Packer) -> None:
        super().__init__(f"The packer '{packer.__class__.__name__}' can't pack clip with alpha plane")
        self.packer = packer


class FramePropsWarning(Warning):
    """Base class for warnings related to VapourSynth frame properties."""


class TransferPropWarning(FramePropsWarning):
    """Warning emitted when the '_Transfer' property is missing or unspecified."""


class PrimariesPropWarning(FramePropsWarning):
    """Warning emitted when the '_Primaries' property is missing or unspecified."""


def _guess_primaries_transfer(matrix: int) -> tuple[IntEnum, IntEnum]:
    match matrix:
        case vs.MATRIX_RGB:
            return vs.PRIMARIES_BT709, vs.TRANSFER_IEC_61966_2_1
        case vs.MATRIX_BT709:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT709
        case vs.MATRIX_BT470_BG:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT601
        case vs.MATRIX_ST170_M:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT601
        case vs.MATRIX_ST240_M:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT601
        case vs.MATRIX_YCGCO:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT709
        case vs.MATRIX_BT2020_CL | vs.MATRIX_BT2020_NCL:
            return vs.PRIMARIES_BT2020, vs.TRANSFER_BT2020_10
        case vs.MATRIX_ICTCP:
            return vs.PRIMARIES_BT709, vs.TRANSFER_BT2020_10
        case _:
            return vs.PRIMARIES_UNSPECIFIED, vs.TRANSFER_UNSPECIFIED


def _normalize_color_props(
    n: int,
    f: vs.VideoFrame,
    props_policy: Literal["warn", "ignore"],
    warned: dict[str, dict[IntEnum, bool]],
) -> vs.VideoFrame:
    if f.props.get("_Matrix", vs.MATRIX_UNSPECIFIED) == vs.MATRIX_UNSPECIFIED:
        match f.format.color_family:
            case vs.YUV:
                # Too ambiguous, this will error in the resize call
                return f
            case vs.GRAY:
                # Silently set the matrix prop to BT709 if format is GRAY
                f = f.copy()
                f.props["_Matrix"] = vs.MATRIX_BT709
            case vs.RGB:
                # Silently set the matrix prop to RGB if format is also RGB
                f = f.copy()
                f.props["_Matrix"] = vs.MATRIX_RGB

    if (
        f.props.get("_Transfer", vs.TRANSFER_UNSPECIFIED) != vs.TRANSFER_UNSPECIFIED
        and f.props.get("_Primaries", vs.PRIMARIES_UNSPECIFIED) != vs.PRIMARIES_UNSPECIFIED
    ):
        return f

    primaries, transfer = _guess_primaries_transfer(f.props["_Matrix"])

    def ensure_prop(prop: str, value: IntEnum, unspecified: IntEnum, category: type[FramePropsWarning]) -> None:
        nonlocal f
        if f.props.get(prop, unspecified) != unspecified:
            return

        if f.readonly:
            f = f.copy()
        f.props[prop] = value

        if props_policy == "warn" and not warned[prop].get(value):
            warned[prop][value] = True
            assumed = value.name.removeprefix(f"{prop[1:].upper()}_")
            warnings.warn(
                f"Unspecified {prop!r} property for frame {n}. Assuming {assumed!r} ({value}) based on '_Matrix'.",
                category=category,
                stacklevel=2,
            )
            logger.log(
                DEBUG - 1,
                "Set %s to %r (%d) based on '_Matrix' (%d)",
                prop,
                assumed,
                value,
                f.props["_Matrix"],
            )

    ensure_prop("_Transfer", transfer, vs.TRANSFER_UNSPECIFIED, TransferPropWarning)
    ensure_prop("_Primaries", primaries, vs.PRIMARIES_UNSPECIFIED, PrimariesPropWarning)

    return f


class Packer(ABC):
    """Abstract base class for RGB packers."""

    FORMAT_CONFIG: Mapping[int, tuple[vs.PresetVideoFormat, vs.PresetVideoFormat, QImage.Format, QImage.Format]] = {
        8: (vs.RGB24, vs.GRAY8, QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32),
        10: (vs.RGB30, vs.GRAY10, QImage.Format.Format_RGB30, QImage.Format.Format_A2RGB30_Premultiplied),
    }

    name: ClassVar[str]

    def __init__(self, bit_depth: int) -> None:
        self.bit_depth = bit_depth
        self.vs_format, self.vs_aformat, self.qt_format, self.qt_aformat = Packer.FORMAT_CONFIG[bit_depth]

    def to_rgb_planar(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        """Converts clip to planar vs.RGB24 or vs.RGB30."""
        if (ppolicy := SettingsManager.global_settings.view.props_policy) != "error":
            warned: dict[str, dict[IntEnum, bool]] = {"_Transfer": {}, "_Primaries": {}}
            clip = clip.std.ModifyFrame(clip, lambda n, f: _normalize_color_props(n, f, ppolicy, warned))

        params = dict[str, Any](
            format=self.vs_format,
            dither_type=SettingsManager.global_settings.view.dither_type,
            resample_filter_uv=SettingsManager.global_settings.view.chroma_resizer.vs_func,
            filter_param_a_uv=SettingsManager.global_settings.view.chroma_resizer.param_a,
            filter_param_b_uv=SettingsManager.global_settings.view.chroma_resizer.param_b,
            transfer=vs.TRANSFER_BT709,
            primaries=vs.PRIMARIES_BT709,
        )
        return clip.resize.Point(**params | kwargs)

    @abstractmethod
    def to_rgb_packed(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        """Converts planar vs.RGB24 or vs.RGB30 to interleaved BGRA32 or RGB30 to packed A2R10G10B10"""

    def pack_clip(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        """Converts a planar VideoNode and an optional alpha mask to a packed RGB/RGBA VideoNode."""
        if isinstance(alpha, vs.VideoNode):
            alpha = alpha.resize.Point(
                format=self.vs_aformat,
                dither_type=SettingsManager.global_settings.view.dither_type,
            )

        planar = self.to_rgb_planar(clip)
        packed = self.to_rgb_packed(planar, alpha)

        return packed.std.SetFrameProp("VSViewHasAlpha", True) if alpha else packed

    def frame_to_qimage(self, frame: vs.VideoFrame, **kwargs: Any) -> QImage:
        """
        Wraps a packed VapourSynth VideoFrame into a QImage.

        If the `copy_qimage` setting is enabled, ownership of the memory is transferred to Qt
        by returning a copy of the image.
        Otherwise, the returned QImage **does not own its memory** and points directly
        to the VapourSynth frame's buffer.

        !!! warning
            When `copy_qimage` is disabled, you MUST either keep the source `frame` alive as long
            as the QImage is used, or call ``.copy()`` on the returned QImage.
        """

        alpha = "VSViewHasAlpha" in frame.props or "_Alpha" in frame.props

        params = dict[str, Any](format=self.qt_aformat if alpha else self.qt_format) | kwargs

        # QImage supports Buffer inputs
        img = QImage(
            get_plane_buffer(frame, 0),  # type: ignore[call-overload]
            frame.width,
            frame.height,
            frame.get_stride(0),
            params.pop("format"),
            **params,
        )

        if SettingsManager.global_settings.view.copy_qimage:
            return img.copy()

        return img


class VszipPacker(Packer):
    name = "vszip"

    def to_rgb_packed(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        if alpha:
            raise AlphaNotImplementedError(self)

        return clip.vszip.PackRGB()


class VSPackRGB(Packer):
    def to_rgb_packed(self, clip: vs.VideoNode, alpha: vs.VideoNode | Literal[True] | None = None) -> vs.VideoNode:
        return packrgb(clip, alpha, self.name)  # type: ignore[arg-type]


class CythonPacker(VSPackRGB):
    name = "cython"


class NumpyPacker(VSPackRGB):
    name = "numpy"


class PythonPacker(VSPackRGB):
    name = "python"


@cache
def _is_vszip_available() -> bool:
    with create_environment(set_logger=False) as env, env.use():
        return hasattr(env.core, "vszip") and hasattr(env.core.vszip, "PackRGB")


def get_packer(method: str | None = None, bit_depth: int | None = None) -> Packer:
    """
    Get the packer to use for packing clips.

    Args:
        method: The packing method to use. If None, the global setting will be used.
        bit_depth: The bit depth to use. If None, the global setting will be used.

    Returns:
        The packer to use for packing clips.
    """
    method = method or SettingsManager.global_settings.view.packing_method
    bit_depth = bit_depth or SettingsManager.global_settings.view.bit_depth

    if method == "auto":
        method = "vszip" if _is_vszip_available() else "cython"
        logger.debug("Auto-selected packing method: %s", method)

    match method:
        case "vszip":
            if not _is_vszip_available():
                logger.warning("vszip plugin is not available, falling back to Cython (8-bit) packer")
                return CythonPacker(8)

            return VszipPacker(bit_depth)

        case "cython":
            return CythonPacker(bit_depth)

        case "numpy":
            return NumpyPacker(bit_depth)

        case "python":
            return PythonPacker(bit_depth)

        case _:
            raise NotImplementedError
