from __future__ import annotations

from logging import getLogger
from typing import Annotated

from pydantic import BaseModel
from vapoursynth import AudioNode, core

from vsview.api import Dropdown, NodeProcessor, Spin, hookimpl

logger = getLogger(__name__)


class GlobalSettings(BaseModel):
    sample_type: Annotated[
        str,
        Dropdown(
            label="Sample type",
            items=[
                ("Integer 16-bit", "i16"),
                ("Integer 24-bit", "i24"),
                ("Integer 32-bit", "i32"),
                ("Float 32-bit", "f32"),
            ],
            tooltip="The sample type to convert to",
        ),
    ] = "f32"
    sample_rate: Annotated[
        int | None,
        Spin(
            label="Sample rate",
            min=-1,
            max=1_000_000,
            suffix=" Hz",
            tooltip="Target sample rate to convert to.\n"
            "-1 means no resampling is performed.\n"
            "If ares is not installed, this option is ignored.",
            to_ui=lambda v: -1 if v is None else v,
            from_ui=lambda v: None if v < 0 else v,
        ),
    ] = None
    quality: Annotated[
        str | None,
        Dropdown(
            label="Quality",
            items=[
                ("Default", None),
                ("Quick", "quick"),
                ("Low", "low"),
                ("Medium", "medium"),
                ("High", "high"),
                ("Very high", "very_high"),
                ("Maximum", "max"),
            ],
            tooltip="The SoX resampler quality. Default is 'very high'.\n"
            "If ares is not installed, this option is ignored.",
        ),
    ] = None


class AudioConvert(NodeProcessor[AudioNode, GlobalSettings]):
    identifier = "jet_vsview_audioconvert"
    display_name = "Audio Convert"

    def prepare(self, audio: AudioNode) -> AudioNode:
        if hasattr(core, "ares"):
            logger.debug("Using ares.Resample on audio %r", audio)
            return audio.ares.Resample(
                sample_rate=self.settings.global_.sample_rate,
                sample_type=self.settings.global_.sample_type,
                quality=self.settings.global_.quality,
            )

        if hasattr(core, "atools"):
            logger.debug("Using atools.Convert on audio %r", audio)
            if self.settings.global_.sample_rate is not None or self.settings.global_.quality is not None:
                logger.warning("Can't set sample rate and/or quality when ares.Resample isn't installed")

            return audio.atools.Convert(sample_type=self.settings.global_.sample_type)

        logger.error("No audio processing method available. Please install ares or atools.")

        return audio


@hookimpl
def vsview_get_audio_processor() -> type[AudioConvert]:
    return AudioConvert
