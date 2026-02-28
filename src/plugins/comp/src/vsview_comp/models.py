from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

from jetpytools import classproperty
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = getLogger(__name__)


class ComparisonImage(NamedTuple):
    path: Path
    pict_type: str
    frame_no: int
    timestamp: str


class ComparisonSource(NamedTuple):
    name: str
    images: Sequence[ComparisonImage]


class TMDBGenre(BaseModel):
    id: int
    name: str


class TMDBProductionCountry(BaseModel):
    iso_3166_1: str = ""


class LoggedBaseModel(BaseModel):
    @classmethod
    def validate_logged(cls, obj: Any, log_context: str) -> Self | None:
        try:
            return super().model_validate(obj)
        except ValidationError as e:
            logger.warning("Ignoring invalid payload in %s: %s", log_context, e)
            logger.debug("Payload validation traceback", exc_info=True)
            return None


class TMDBTitleData(LoggedBaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    name: str = ""
    title: str = ""
    original_name: str | None = None
    original_title: str | None = None
    original_language: str | None = None
    origin_country: list[str] = Field(default_factory=list)
    production_countries: list[TMDBProductionCountry] = Field(default_factory=list)
    first_air_date: str | None = None
    release_date: str | None = None
    poster_path: str | None = None
    overview: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    genre_ids: list[int] = Field(default_factory=list)
    genres: list[TMDBGenre] = Field(default_factory=list)


class TMDBPayload(LoggedBaseModel):
    results: list[TMDBTitleData] = Field(default_factory=list)
    genres: list[TMDBGenre] = Field(default_factory=list)


class TMDBTooltipData(BaseModel):
    header: str
    original_name: str
    genres: str
    language: str
    country: str
    release_date: str
    rating: str
    popularity: str
    tmdb_id: str
    overview: str


class TMDBTitle(BaseModel):
    model_config = ConfigDict(ignored_types=(classproperty,))

    data: TMDBTitleData
    media_type: Literal["tv", "movie"]
    genres: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """TMDB ID"""
        return str(self.data.id)

    @property
    def name(self) -> str:
        """Title name"""
        return self.data.name if self.media_type == "tv" else self.data.title

    @property
    def original_name(self) -> str:
        """Original title name"""
        return self.data.original_name or self.data.original_title or "-"

    @property
    def year(self) -> str:
        """Release year"""
        release_date = self.data.first_air_date if self.media_type == "tv" else self.data.release_date
        return (release_date or "0000")[:4]

    @property
    def media_tag(self) -> str:
        """Media tag (TV or Movie)"""
        return self.media_type.upper()

    @property
    def language(self) -> str:
        """Original language"""
        return (self.data.original_language or "-").upper()

    @property
    def country(self) -> str:
        """Original country"""
        return (
            ", ".join(
                [c for c in self.data.origin_country if c]
                or [c.iso_3166_1 for c in self.data.production_countries if c.iso_3166_1]
            )
            or "-"
        )

    @property
    def release_date(self) -> str:
        """Release date"""
        return (self.data.first_air_date or self.data.release_date or "-").strip()

    @property
    def rating(self) -> str:
        """Rating"""
        if self.data.vote_average is not None and self.data.vote_count is not None and self.data.vote_count > 0:
            return f"{self.data.vote_average:.1f} ({self.data.vote_count} votes)"
        else:
            return "Not enough ratings"

    @property
    def popularity(self) -> str:
        """Popularity"""
        return f"{self.data.popularity:.2f}" if self.data.popularity is not None else "-"

    @property
    def overview(self) -> str:
        """Overview"""
        return (self.data.overview or "-").strip()

    @property
    def tooltip_data(self) -> TMDBTooltipData:
        overview = self.overview
        if len(overview) > 480:
            overview = f"{overview[:477].rstrip()}..."

        return TMDBTooltipData(
            header=f"{self.name} ({self.year}) [{self.media_tag}]",
            original_name=self.original_name,
            genres=", ".join(self.genres) if self.genres else "-",
            language=self.language,
            country=self.country,
            release_date=self.release_date,
            rating=self.rating,
            popularity=self.popularity,
            tmdb_id=self.id,
            overview=overview,
        )

    @classproperty
    @classmethod
    def supported_format_fields(cls) -> tuple[str, ...]:
        return ("id", "name", "original_name", "year", "media_tag", "language", "country", "release_date")

    @classproperty
    @classmethod
    def format_hints(cls) -> dict[str, str]:
        hints = dict[str, str]()
        for field in cls.supported_format_fields:
            if not isinstance((prop := getattr(cls, field, None)), property):
                logger.warning("Field %s is not a property", field)
                continue

            hints[field] = (prop.fget.__doc__ or "").strip()
        return hints

    def format_name(self, fmt: str, **extra: str) -> str:
        """Format a name string using available TMDB fields and extra context."""

        return fmt.format_map(
            defaultdict(
                str,
                **{field: str(getattr(self, field)) for field in self.supported_format_fields},
                **extra,
            )
        )
