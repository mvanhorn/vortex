# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Resolve declarative benchmark definitions into GitHub Actions matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from .config import Benchmark, BenchmarkTarget, Engine, Format


class Storage(Enum):
    """Where a benchmark's data lives when it runs."""

    NVME = "nvme"
    S3 = "s3"

    @property
    def label(self) -> str:
        """Human-facing name used in benchmark display names."""
        return "NVME" if self is Storage.NVME else "S3"


def _dedupe(targets: Iterable[BenchmarkTarget]) -> tuple[BenchmarkTarget, ...]:
    """Normalize and de-duplicate targets, preserving first-seen order."""
    return tuple(dict.fromkeys(target.normalized() for target in targets))


@dataclass(frozen=True)
class TargetSet:
    """An ordered set of engine/format targets with small set algebra."""

    targets: tuple[BenchmarkTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _dedupe(self.targets))

    def __or__(self, other: TargetSet) -> TargetSet:
        """Return the ordered union of two target sets."""
        return TargetSet(self.targets + other.targets)

    def only(self, *engines: Engine) -> TargetSet:
        """Restrict the target set to the given engines."""
        keep = set(engines)
        return TargetSet(tuple(target for target in self.targets if target.engine in keep))

    def formats(self) -> list[Format]:
        """Return referenced formats in first-seen order."""
        return list(dict.fromkeys(target.format for target in self.targets))

    def __iter__(self):
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)


def targets(engine: Engine, *formats: Format) -> TargetSet:
    """Build targets for one engine across several formats."""
    return TargetSet(tuple(BenchmarkTarget(engine=engine, format=fmt) for fmt in formats))


def df(*formats: Format) -> TargetSet:
    """Build DataFusion targets across several formats."""
    return targets(Engine.DATAFUSION, *formats)


def duck(*formats: Format) -> TargetSet:
    """Build DuckDB targets across several formats."""
    return targets(Engine.DUCKDB, *formats)


STANDARD = df(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT) | duck(
    Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT
)
DEFAULTS = df(Format.PARQUET, Format.VORTEX) | duck(Format.PARQUET, Format.VORTEX)
_NOT_GENERATED = frozenset({Format.ARROW, Format.LANCE})
_FORMAT_ORDER = (
    Format.ARROW,
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.VORTEX_NATIVE,
    Format.DUCKDB,
    Format.LANCE,
)


@dataclass(frozen=True)
class BenchmarkDef:
    """A benchmark and the canonical target superset it can run."""

    id: str
    benchmark: Benchmark
    name: str
    targets: TargetSet
    storage: Storage = Storage.NVME
    scale_factor: float | int | None = None
    iterations: int | None = None
    nightly: bool = False
    local_dir: str | None = None
    remote_storage: str | None = None

    @property
    def subcommand(self) -> str:
        """Return the ``vx-bench`` subcommand for this benchmark."""
        return self.benchmark.value


TargetPolicy = Callable[[BenchmarkDef], TargetSet]


def all_targets(benchmark: BenchmarkDef) -> TargetSet:
    """Run every target declared by a benchmark."""
    return benchmark.targets


def defaults(benchmark: BenchmarkDef) -> TargetSet:
    """Run the cheap default lane intersected with a benchmark's declared targets."""
    return TargetSet(tuple(target for target in benchmark.targets if target in DEFAULTS.targets))


@dataclass(frozen=True)
class Profile:
    """A named CI benchmark configuration."""

    nightly: bool = False
    targets: TargetPolicy = all_targets
    description: str = ""


def _valid_for_storage(target_set: TargetSet, storage: Storage) -> TargetSet:
    """Drop targets that are invalid for the storage backend."""
    if storage is Storage.S3:
        return TargetSet(tuple(target for target in target_set if target.format is not Format.LANCE))
    return target_set


def _data_formats(target_set: TargetSet) -> list[Format]:
    """Return data formats that the data-generation step must produce."""
    present = set(target_set.formats())
    return [fmt for fmt in _FORMAT_ORDER if fmt in present and fmt not in _NOT_GENERATED]


def _matrix_entry(benchmark: BenchmarkDef, run_targets: TargetSet) -> dict[str, object]:
    """Build one GitHub Actions ``include`` entry."""
    entry: dict[str, object] = {
        "id": benchmark.id,
        "subcommand": benchmark.subcommand,
        "name": benchmark.name,
        "targets": [target.to_dict() for target in run_targets],
        "data_formats": [fmt.value for fmt in _data_formats(run_targets)],
    }
    if benchmark.scale_factor is not None:
        entry["scale_factor"] = str(benchmark.scale_factor)
    if benchmark.iterations is not None:
        entry["iterations"] = str(benchmark.iterations)
    if benchmark.local_dir is not None:
        entry["local_dir"] = benchmark.local_dir
    if benchmark.remote_storage is not None:
        entry["remote_storage"] = benchmark.remote_storage
    return entry


def resolve_matrix(profile: Profile, benchmarks: Iterable[BenchmarkDef]) -> list[dict[str, object]]:
    """Resolve a profile into GitHub Actions matrix entries."""
    entries: list[dict[str, object]] = []
    for benchmark in benchmarks:
        if benchmark.nightly != profile.nightly:
            continue
        run_targets = _valid_for_storage(profile.targets(benchmark), benchmark.storage)
        if len(run_targets) == 0:
            continue
        entries.append(_matrix_entry(benchmark, run_targets))
    return entries
