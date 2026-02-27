from logging import getLogger
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = getLogger(__name__)


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
    data: TMDBTitleData
    media_type: Literal["tv", "movie"]
    genres: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.data.id)

    @property
    def name(self) -> str:
        return self.data.name if self.media_type == "tv" else self.data.title

    @property
    def year(self) -> str:
        release_date = self.data.first_air_date if self.media_type == "tv" else self.data.release_date
        return (release_date or "0000")[:4]

    @property
    def media_tag(self) -> str:
        return self.media_type.upper()

    @property
    def poster_path(self) -> str | None:
        return self.data.poster_path

    @property
    def tooltip_data(self) -> TMDBTooltipData:
        data = self.data
        original_name = data.original_name or data.original_title or "-"
        language = (data.original_language or "-").upper()
        countries = [c for c in data.origin_country if c] or [
            c.iso_3166_1 for c in data.production_countries if c.iso_3166_1
        ]
        country = ", ".join(countries) or "-"
        release_date = (data.first_air_date or data.release_date or "-").strip()

        if data.vote_average is not None and data.vote_count is not None and data.vote_count > 0:
            rating = f"{data.vote_average:.1f} ({data.vote_count} votes)"
        else:
            rating = "Not enough ratings"

        popularity = f"{data.popularity:.2f}" if data.popularity is not None else "-"
        overview = (data.overview or "-").strip()
        if len(overview) > 480:
            overview = f"{overview[:477].rstrip()}..."

        return TMDBTooltipData(
            header=f"{self.name} ({self.year}) [{self.media_tag}]",
            original_name=original_name,
            genres=", ".join(self.genres) if self.genres else "-",
            language=language,
            country=country,
            release_date=release_date,
            rating=rating,
            popularity=popularity,
            tmdb_id=self.id,
            overview=overview,
        )
