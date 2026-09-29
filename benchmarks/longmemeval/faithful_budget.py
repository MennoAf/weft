#!/usr/bin/env python3
"""faithful_budget.py — Conservative, hash-bound budget ledger for S36.

This module records durable reservations before network attempts. It never
contacts a provider and is safe to use from concurrent runner processes. An
unknown outcome retains its entire reservation, so a caller cannot silently
replay an ambiguous attempt.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-09-21
Python:  >= 3.12

Dependencies:
    (stdlib only) — locking, JSON, hashing, and atomic file replacement.

Usage:
    Imported by ``faithful_agent.py`` and ``faithful_s36.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_TOTAL_BUDGET_USD = 20.0
GPT6_TOTAL_BUDGET_USD = 50.0
DEFAULT_CALIBRATION_BUDGET_USD = 5.0
BUDGET_LEDGER_SCHEMA = "weft.longmemeval.faithful-budget.v3"
MILLION = 1_000_000.0
GPT6_LONG_CONTEXT_THRESHOLD_TOKENS = 272_000


class BudgetError(RuntimeError):
    """Base class for budget-ledger failures."""


class BudgetExceeded(BudgetError):
    """Raised before an attempt that would exceed a configured budget ceiling."""


class TotalBudgetExceeded(BudgetExceeded):
    """Raised before a request that would reach or exceed the total run ceiling."""

    def __init__(self, message: str, *, used_usd: float, estimate_usd: float, ceiling_usd: float) -> None:
        super().__init__(message)
        self.used_usd = used_usd
        self.estimate_usd = estimate_usd
        self.ceiling_usd = ceiling_usd


class LedgerBindingError(BudgetError):
    """Raised when a ledger is reused for different benchmark artifacts."""


class InvalidUsage(BudgetError):
    """Raised when provider usage is absent, negative, or non-finite."""


@dataclass(frozen=True, slots=True)
class Pricing:
    """Legacy per-million-token prices; keep its persisted shape stable."""

    luna_input_usd_per_million: float = 0.40
    luna_cache_write_usd_per_million: float = 0.50
    luna_output_usd_per_million: float = 1.80
    gpt4o_input_usd_per_million: float = 2.50
    gpt4o_output_usd_per_million: float = 10.00

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("pricing values must be finite and non-negative")

    def reserve_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate an upper-bound reservation using the legacy price table."""
        _validate_count(input_tokens, "input_tokens")
        _validate_count(output_tokens, "output_tokens")
        lowered = model.lower()
        if "luna" in lowered:
            input_rate = self.luna_input_usd_per_million + self.luna_cache_write_usd_per_million
            output_rate = self.luna_output_usd_per_million
        elif "gpt-4o" in lowered or "gpt4o" in lowered:
            input_rate = self.gpt4o_input_usd_per_million
            output_rate = self.gpt4o_output_usd_per_million
        else:
            raise ValueError(f"unsupported budget model: {model!r}")
        return (input_tokens * input_rate + output_tokens * output_rate) / MILLION

    def actual_cost(self, model: str, usage: Mapping[str, Any]) -> float:
        """Calculate legacy actual cost without cache discounts."""
        validated = validate_usage(usage)
        lowered = model.lower()
        if "luna" in lowered:
            input_rate = self.luna_input_usd_per_million
            output_rate = self.luna_output_usd_per_million
        elif "gpt-4o" in lowered or "gpt4o" in lowered:
            input_rate = self.gpt4o_input_usd_per_million
            output_rate = self.gpt4o_output_usd_per_million
        else:
            raise ValueError(f"unsupported budget model: {model!r}")
        return (validated["input_tokens"] * input_rate + validated["output_tokens"] * output_rate) / MILLION


@dataclass(frozen=True, slots=True)
class FreshRunPricing:
    """GPT-6/GPT-4o pricing for newly prepared selected-35 runs only."""

    gpt6_input_usd_per_million: float = 0.10
    gpt6_cached_input_usd_per_million: float = 0.01
    gpt6_cache_write_usd_per_million: float = 0.125
    gpt6_output_usd_per_million: float = 0.50
    gpt6_long_input_usd_per_million: float = 0.20
    gpt6_long_cached_input_usd_per_million: float = 0.02
    gpt6_long_cache_write_usd_per_million: float = 0.25
    gpt6_long_output_usd_per_million: float = 0.75
    gpt4o_input_usd_per_million: float = 2.50
    gpt4o_cached_input_usd_per_million: float = 1.25
    gpt4o_output_usd_per_million: float = 10.00
    long_context_threshold_tokens: int = GPT6_LONG_CONTEXT_THRESHOLD_TOKENS

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "long_context_threshold_tokens":
                _validate_count(value, name)
            elif not math.isfinite(value) or value < 0:
                raise ValueError("pricing values must be finite and non-negative")

    def reserve_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Reserve with worst-case cache-write pricing for every input token."""
        _validate_count(input_tokens, "input_tokens")
        _validate_count(output_tokens, "output_tokens")
        input_rate, output_rate = self._rates(model, input_tokens)
        if "gpt-4o" in model.lower() or "gpt4o" in model.lower():
            input_rate = self.gpt4o_input_usd_per_million
        elif "gpt-6" in model.lower() and "luna" in model.lower():
            input_rate = (self.gpt6_long_cache_write_usd_per_million
                          if input_tokens > self.long_context_threshold_tokens
                          else self.gpt6_cache_write_usd_per_million)
        else:
            raise ValueError(f"unsupported fresh-run budget model: {model!r}")
        return (input_tokens * input_rate + output_tokens * output_rate) / MILLION

    def actual_cost(self, model: str, usage: Mapping[str, Any]) -> float:
        """Price measured cache reads/writes; unknown writes use worst-case rates."""
        validated = validate_usage(usage)
        input_tokens = validated["input_tokens"]
        output_tokens = validated["output_tokens"]
        lowered = model.lower()
        if "gpt-4o" in lowered or "gpt4o" in lowered:
            cached = validated["cached_input_tokens"]
            cached_tokens = 0 if cached is None else cached
            if cached_tokens > input_tokens:
                raise InvalidUsage("cached_input_tokens cannot exceed input_tokens")
            input_cost = (
                cached_tokens * self.gpt4o_cached_input_usd_per_million
                + (input_tokens - cached_tokens) * self.gpt4o_input_usd_per_million
            )
            return (input_cost + output_tokens * self.gpt4o_output_usd_per_million) / MILLION
        if "gpt-6" not in lowered or "luna" not in lowered:
            raise ValueError(f"unsupported fresh-run budget model: {model!r}")

        long_context = input_tokens > self.long_context_threshold_tokens
        cached_rate = (self.gpt6_long_cached_input_usd_per_million if long_context
                       else self.gpt6_cached_input_usd_per_million)
        uncached_rate = (self.gpt6_long_input_usd_per_million if long_context
                         else self.gpt6_input_usd_per_million)
        cache_write_rate = (self.gpt6_long_cache_write_usd_per_million if long_context
                            else self.gpt6_cache_write_usd_per_million)
        cached = validated["cached_input_tokens"]
        cache_write = validated["cache_write_input_tokens"]
        # Without a measured cache-read split, count all input at the higher
        # write rate. Do not add cached input on top of input_tokens: it is a
        # subset of the total, never an additional token volume.
        if cached is None:
            input_cost = input_tokens * cache_write_rate
        else:
            if cached > input_tokens:
                raise InvalidUsage("cached_input_tokens cannot exceed input_tokens")
            remaining = input_tokens - cached
            if cache_write is None:
                write_tokens = remaining
            else:
                if cache_write > remaining:
                    raise InvalidUsage("cache_write_input_tokens exceeds uncached input tokens")
                write_tokens = cache_write
            uncached_tokens = remaining - write_tokens
            input_cost = (
                cached * cached_rate
                + uncached_tokens * uncached_rate
                + write_tokens * cache_write_rate
            )
        return (input_cost + output_tokens * (self.gpt6_long_output_usd_per_million if long_context
                                               else self.gpt6_output_usd_per_million)) / MILLION

    def _rates(self, model: str, input_tokens: int) -> tuple[float, float]:
        """Return output rate and a valid input rate for validation dispatch."""
        lowered = model.lower()
        if "gpt-4o" in lowered or "gpt4o" in lowered:
            return self.gpt4o_input_usd_per_million, self.gpt4o_output_usd_per_million
        if "gpt-6" in lowered and "luna" in lowered:
            return self.gpt6_input_usd_per_million, (
                self.gpt6_long_output_usd_per_million if input_tokens > self.long_context_threshold_tokens
                else self.gpt6_output_usd_per_million
            )
        raise ValueError(f"unsupported fresh-run budget model: {model!r}")


def _validate_count(value: Any, name: str) -> int:
    """Validate a non-negative integer token count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidUsage(f"{name} must be a non-negative integer")
    return value


def validate_usage(usage: Mapping[str, Any]) -> dict[str, int | None]:
    """Validate required totals and preserve absent cache splits as unknown.

    Cached/write counts partition ``input_tokens``; they are never added on top
    of it. Missing cache details remain ``None`` so FreshRunPricing can price
    the uncertain volume conservatively rather than treating it as zero.
    """
    if not isinstance(usage, Mapping):
        raise InvalidUsage("provider usage must be a mapping")
    if "input_tokens" not in usage or "output_tokens" not in usage:
        raise InvalidUsage("provider usage must include input_tokens and output_tokens")
    result: dict[str, int | None] = {
        "input_tokens": _validate_count(usage.get("input_tokens"), "input_tokens"),
        "output_tokens": _validate_count(usage.get("output_tokens"), "output_tokens"),
    }
    for key in ("cached_input_tokens", "cache_write_input_tokens"):
        raw = usage.get(key)
        result[key] = None if raw is None else _validate_count(raw, key)
    result["reasoning_tokens"] = _validate_count(usage.get("reasoning_tokens", 0), "reasoning_tokens")
    return result


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible data deterministically."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _money(value: Any, name: str) -> float:
    """Validate one persisted monetary value used for fail-closed accounting."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise BudgetError(f"{name} must be a finite non-negative number")
    return value


def _reservation_charge(row: Mapping[str, Any]) -> float:
    """Charge the conservative maximum of estimate and observed usage."""
    estimate = _money(row.get("estimated_usd", 0.0), "estimated_usd")
    actual = row.get("actual_usd")
    return estimate if actual is None else max(estimate, _money(actual, "actual_usd"))


def import_prior_ledger(path: Path) -> dict[str, Any]:
    """Import one immutable ledger, verifying and flattening any v3 chain.

    A v3 carry-forward stores a flattened predecessor reservation list. Before
    trusting it, recursively verify the linked predecessor's path, byte hash,
    aggregate totals, and normalized reservation rows. Unknown charges remain
    their original estimates through :func:`_reservation_charge`.
    """
    def import_chain(candidate: Path, ancestors: frozenset[Path]) -> dict[str, Any]:
        candidate = Path(candidate)
        resolved = candidate.resolve()
        if resolved in ancestors:
            raise LedgerBindingError("prior ledger carry-forward chain contains a cycle")
        if not resolved.is_file():
            raise LedgerBindingError(f"prior ledger is missing: {resolved}")
        source_sha256 = sha256_file(resolved)
        path_digest = canonical_hash(str(resolved))
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerBindingError(f"prior ledger is unreadable: {resolved}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("reservations"), list):
            raise LedgerBindingError("prior ledger must contain a reservations list")

        schema = data.get("schema")
        inherited: list[dict[str, Any]] = []
        carry = data.get("carry_forward", {})
        if schema == BUDGET_LEDGER_SCHEMA:
            if not isinstance(carry, Mapping):
                raise LedgerBindingError("v3 carry_forward must be an object")
            if not carry:
                raise LedgerBindingError("v3 ledger lacks a verifiable carry-forward predecessor")
            if carry:
                source_path = carry.get("source_path")
                source_hash = carry.get("source_sha256")
                source_digest = carry.get("source_path_digest")
                if not all(isinstance(value, str) and value for value in (source_path, source_hash, source_digest)):
                    raise LedgerBindingError("v3 carry_forward predecessor identity is incomplete")
                predecessor = import_chain(Path(source_path), ancestors | {resolved})
                if predecessor["source_path"] != source_path:
                    raise LedgerBindingError("v3 carry_forward predecessor path identity mismatch")
                if predecessor["source_sha256"] != source_hash:
                    raise LedgerBindingError("v3 carry_forward predecessor hash mismatch")
                if predecessor["source_path_digest"] != source_digest:
                    raise LedgerBindingError("v3 carry_forward predecessor path digest mismatch")
                count = carry.get("reservation_count")
                if isinstance(count, bool) or not isinstance(count, int) or count != predecessor["reservation_count"]:
                    raise LedgerBindingError("v3 carry_forward reservation count mismatch")
                for field in ("imported_usd", "imported_calibration_usd"):
                    amount = _money(carry.get(field), f"carry_forward.{field}")
                    if not math.isclose(amount, predecessor[field], rel_tol=1e-12, abs_tol=1e-12):
                        raise LedgerBindingError(f"v3 carry_forward {field} mismatch")
                rows = carry.get("reservations")
                if not isinstance(rows, list) or rows != predecessor["reservations"]:
                    raise LedgerBindingError("v3 carry_forward reservation rows mismatch")
                inherited = predecessor["reservations"]
        elif carry:
            raise LedgerBindingError("prior ledger already contains a carry-forward import")
        elif not isinstance(carry, Mapping):
            raise LedgerBindingError("prior ledger carry_forward must be an object")

        imported = list(inherited)
        seen = {row["reservation_id"] for row in imported}
        total = sum(row["charge_usd"] for row in imported)
        calibration_total = sum(
            row["charge_usd"] for row in imported if row["phase"] == "calibration"
        )
        for row in data["reservations"]:
            if not isinstance(row, Mapping):
                raise LedgerBindingError("prior ledger contains a malformed reservation")
            reservation_id = row.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id or reservation_id in seen:
                raise LedgerBindingError("prior ledger contains duplicate or invalid reservation IDs")
            seen.add(reservation_id)
            charge = _reservation_charge(row)
            phase = row.get("phase", "run")
            if phase not in {"calibration", "run"}:
                raise LedgerBindingError("prior ledger contains an invalid reservation phase")
            imported_row = {
                "reservation_id": reservation_id,
                "model": str(row.get("model", "")),
                "phase": phase,
                "charge_usd": charge,
            }
            imported.append(imported_row)
            total += charge
            if phase == "calibration":
                calibration_total += charge
        return {
            "source_path": str(resolved),
            "source_sha256": source_sha256,
            "source_path_digest": path_digest,
            "source_schema": schema,
            "reservation_count": len(imported),
            "imported_usd": total,
            "imported_calibration_usd": calibration_total,
            "reservations": imported,
        }

    return import_chain(Path(path), frozenset())


@dataclass(frozen=True, slots=True)
class Reservation:
    """One durable reservation made before a network attempt."""

    reservation_id: str
    model: str
    estimated_usd: float
    phase: str = "run"
    status: str = "reserved"
    actual_usd: float | None = None
    error: str | None = None


class BudgetLedger:
    """Atomic JSON ledger enforcing a cumulative ceiling and artifact binding."""

    def __init__(
        self,
        path: Path,
        *,
        max_budget_usd: float = DEFAULT_TOTAL_BUDGET_USD,
        calibration_budget_usd: float = DEFAULT_CALIBRATION_BUDGET_USD,
        operational_stop_usd: float | None = None,
        pricing: Pricing | None = None,
        binding: Mapping[str, str] | None = None,
        carry_forward: Mapping[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(max_budget_usd) or max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be finite and positive")
        if not math.isfinite(calibration_budget_usd) or calibration_budget_usd <= 0:
            raise ValueError("calibration_budget_usd must be finite and positive")
        effective_operational_stop = max_budget_usd if operational_stop_usd is None else operational_stop_usd
        if (
            isinstance(effective_operational_stop, bool)
            or not isinstance(effective_operational_stop, (int, float))
            or not math.isfinite(effective_operational_stop)
            or effective_operational_stop <= 0
            or effective_operational_stop > max_budget_usd
        ):
            raise ValueError("operational_stop_usd must be finite, positive, and no greater than max_budget_usd")
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.max_budget_usd = max_budget_usd
        self.calibration_budget_usd = calibration_budget_usd
        self.operational_stop_usd = float(effective_operational_stop)
        self.pricing = pricing or Pricing()
        self.binding = dict(binding or {})
        self.carry_forward = dict(carry_forward or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            data = self._read_unlocked()
            if data is None:
                self._write_unlocked(self._new_data())
            else:
                self._check_binding(data)

    def _new_data(self) -> dict[str, Any]:
        """Build the initial ledger document."""
        return {
            "schema": BUDGET_LEDGER_SCHEMA,
            "max_budget_usd": self.max_budget_usd,
            "calibration_budget_usd": self.calibration_budget_usd,
            "operational_stop_usd": self.operational_stop_usd,
            "binding": self.binding,
            "pricing": asdict(self.pricing),
            "carry_forward": self.carry_forward,
            "reservations": [],
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive process lock around a read-modify-write."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any] | None:
        """Read the ledger, returning None for a not-yet-created file."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise BudgetError("ledger must contain a JSON object")
        return value

    def _write_unlocked(self, value: Mapping[str, Any]) -> None:
        """Atomically replace the ledger after validation by the caller."""
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _check_binding(self, data: Mapping[str, Any]) -> None:
        """Reject a ledger whose manifest/checkpoint binding changed."""
        existing = data.get("binding", {})
        if existing != self.binding:
            raise LedgerBindingError(
                f"ledger binding mismatch: existing={existing!r}, requested={self.binding!r}"
            )
        existing_budget = float(data.get("max_budget_usd", self.max_budget_usd))
        if existing_budget != self.max_budget_usd:
            raise LedgerBindingError("ledger budget ceiling cannot change on resume")
        if float(data.get("calibration_budget_usd", self.calibration_budget_usd)) != self.calibration_budget_usd:
            raise LedgerBindingError("ledger calibration ceiling cannot change on resume")
        if float(data.get("operational_stop_usd", self.operational_stop_usd)) != self.operational_stop_usd:
            raise LedgerBindingError("ledger operational stop cannot change on resume")
        if data.get("pricing") != asdict(self.pricing):
            raise LedgerBindingError("ledger pricing snapshot cannot change on resume")
        if data.get("carry_forward", {}) != self.carry_forward:
            raise LedgerBindingError("ledger carry-forward binding cannot change on resume")

    def reserve(self, model: str, input_tokens: int, output_tokens: int, *, phase: str = "run") -> Reservation:
        """Atomically reserve cost before one provider attempt.

        ``phase`` is part of the durable reservation so calibration and the
        later run share one cumulative ledger while calibration is capped at
        the profile-bound sub-ceiling. Reservations are admitted before any
        provider request and are never released on ambiguous outcomes.
        """
        if phase not in {"calibration", "run"}:
            raise ValueError("phase must be calibration or run")
        estimate = self.pricing.reserve_cost(model, input_tokens, output_tokens)
        with self._locked():
            data = self._read_unlocked() or self._new_data()
            self._check_binding(data)
            rows = data["reservations"]
            imported_used = _money(self.carry_forward.get("imported_usd", 0.0), "imported_usd")
            imported_calibration = _money(
                self.carry_forward.get("imported_calibration_usd", 0.0),
                "imported_calibration_usd",
            )
            used = imported_used + sum(_reservation_charge(row) for row in rows)
            calibration_used = imported_calibration + sum(
                _reservation_charge(row)
                for row in rows if row.get("phase", "run") == "calibration"
            )
            stop_ceiling = self.max_budget_usd if phase == "calibration" else self.operational_stop_usd
            if used >= stop_ceiling - 1e-12 or used + estimate >= stop_ceiling - 1e-12:
                raise TotalBudgetExceeded(
                    f"reservation ${estimate:.6f} would reach or exceed the stop ceiling; "
                    f"estimated spend is ${used:.6f} of ${stop_ceiling:.6f}",
                    used_usd=used, estimate_usd=estimate, ceiling_usd=stop_ceiling,
                )
            if phase == "calibration" and calibration_used + estimate > self.calibration_budget_usd + 1e-12:
                raise BudgetExceeded(
                    "calibration reservation would exceed the "
                    f"${self.calibration_budget_usd:.2f} sub-ceiling"
                )
            reservation = Reservation(str(uuid.uuid4()), model, estimate, phase=phase)
            rows.append(asdict(reservation))
            self._write_unlocked(data)
            return reservation

    def finalize(
        self,
        reservation_id: str,
        *,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        unknown: bool = False,
    ) -> Reservation:
        """Finalize a reservation without releasing its conservative reserve.

        ``unknown=True`` is the fail-closed path for timeouts and ambiguous
        writes. The reservation remains fully counted and cannot be replayed.
        """
        with self._locked():
            data = self._read_unlocked() or self._new_data()
            self._check_binding(data)
            for row in data["reservations"]:
                if row["reservation_id"] != reservation_id:
                    continue
                if row["status"] != "reserved":
                    raise BudgetError(f"reservation already finalized: {reservation_id}")
                actual = None if usage is None else self.pricing.actual_cost(row["model"], usage)
                row["status"] = "unknown" if unknown or usage is None else "completed"
                row["actual_usd"] = actual
                row["error"] = error
                self._write_unlocked(data)
                return Reservation(**row)
        raise BudgetError(f"unknown reservation: {reservation_id}")

    @staticmethod
    def _validate_unknown_batch(
        data: Mapping[str, Any],
        reservation_ids: Sequence[str],
        *,
        expected_model: str,
        error: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Validate a fail-closed recovery set; return rows and whether to change them."""
        if not isinstance(expected_model, str) or not expected_model:
            raise ValueError("expected_model is required")
        if not isinstance(error, str) or not error:
            raise ValueError("error reason is required")
        if isinstance(reservation_ids, (str, bytes)) or not isinstance(reservation_ids, Sequence):
            raise ValueError("reservation_ids must be a sequence")
        ids = list(reservation_ids)
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("reservation_ids must contain non-empty IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("reservation_ids must be unique")
        rows = data.get("reservations")
        if not isinstance(rows, list):
            raise BudgetError("ledger reservations must be a list")
        indexed = {
            row.get("reservation_id"): row for row in rows
            if isinstance(row, dict)
        }
        if len(indexed) != len(rows):
            raise BudgetError("ledger contains malformed or duplicate reservation rows")
        selected = [indexed.get(item) for item in ids]
        if any(row is None for row in selected):
            raise BudgetError("recovery reservation ID not found")
        selected_rows = [row for row in selected if row is not None]
        if any(
            row.get("model") != expected_model or row.get("phase") != "run"
            for row in selected_rows
        ):
            raise BudgetError("recovery reservation model or phase mismatch")
        reserved_ids = {
            row.get("reservation_id") for row in rows
            if isinstance(row, dict) and row.get("status") == "reserved"
        }
        expected_ids = set(ids)
        if reserved_ids and reserved_ids != expected_ids:
            raise BudgetError("outstanding reserved IDs differ from recovery set")
        if reserved_ids:
            row_ids = [row.get("reservation_id") for row in rows]
            selected_indexes = [row_ids.index(item) for item in ids]
            expected_suffix = list(range(len(rows) - len(ids), len(rows)))
            if selected_indexes != expected_suffix:
                raise BudgetError("recovery reservations are not the ledger suffix in ledger order")
            if any(row.get("status") != "reserved" for row in selected_rows):
                raise BudgetError("recovery set contains non-reserved rows")
            return selected_rows, True
        if all(row.get("status") == "unknown" and row.get("error") == error for row in selected_rows):
            return selected_rows, False
        raise BudgetError("recovery set is not entirely outstanding or identically recovered")

    def inspect_reservation(self, reservation_id: str) -> dict[str, Any]:
        """Return one reservation row under the ledger lock without mutation."""
        if not isinstance(reservation_id, str) or not reservation_id:
            raise ValueError("reservation_id is required")
        with self._locked():
            data = self._read_unlocked()
            if data is None:
                raise BudgetError("cannot inspect a reservation in an empty ledger")
            self._check_binding(data)
            rows = data.get("reservations")
            if not isinstance(rows, list):
                raise BudgetError("ledger reservations must be a list")
            matches = [
                row for row in rows
                if isinstance(row, dict) and row.get("reservation_id") == reservation_id
            ]
            if len(matches) != 1:
                raise BudgetError("reservation ID must identify exactly one ledger row")
            return dict(matches[0])

    def inspect_prefinalized_timeout_reservation(
        self,
        reservation_id: str,
        *,
        expected_model: str,
        error: str,
    ) -> dict[str, Any]:
        """Verify one provider-timeout unknown row without changing the ledger."""
        if not isinstance(reservation_id, str) or not reservation_id:
            raise ValueError("reservation_id is required")
        if not isinstance(expected_model, str) or not expected_model:
            raise ValueError("expected_model is required")
        if error != "Request timed out.":
            raise ValueError("prefinalized timeout recovery requires the exact timeout error")
        with self._locked():
            data = self._read_unlocked()
            if data is None:
                raise BudgetError("cannot inspect a prefinalized reservation in an empty ledger")
            self._check_binding(data)
            rows = data.get("reservations")
            if not isinstance(rows, list):
                raise BudgetError("ledger reservations must be a list")
            indexed = {
                row.get("reservation_id"): row for row in rows
                if isinstance(row, dict)
            }
            if len(indexed) != len(rows):
                raise BudgetError("ledger contains malformed or duplicate reservation rows")
            row = indexed.get(reservation_id)
            if row is None:
                raise BudgetError("recovery reservation ID not found")
            if row.get("model") != expected_model or row.get("phase") != "run":
                raise BudgetError("recovery reservation model or phase mismatch")
            estimate = row.get("estimated_usd")
            if (
                row.get("status") != "unknown"
                or row.get("error") != error
                or row.get("actual_usd") is not None
                or isinstance(estimate, bool)
                or not isinstance(estimate, (int, float))
                or not math.isfinite(estimate)
                or estimate <= 0
            ):
                raise BudgetError(
                    "reservation is not finalized unknown with the exact timeout and retained estimate"
                )
            if any(
                isinstance(candidate, dict) and candidate.get("status") == "reserved"
                for candidate in rows
            ):
                raise BudgetError("pre-finalized timeout recovery requires zero reserved ledger rows")
            return dict(row)

    def inspect_unknown_batch(
        self,
        reservation_ids: Sequence[str],
        *,
        expected_model: str,
        error: str,
    ) -> dict[str, Any]:
        """Validate a recovery batch without changing the ledger."""
        with self._locked():
            data = self._read_unlocked()
            if data is None:
                raise BudgetError("cannot recover reservations in an empty ledger")
            self._check_binding(data)
            _, changed = self._validate_unknown_batch(
                data, reservation_ids, expected_model=expected_model, error=error,
            )
        return {
            "reservation_ids": list(reservation_ids),
            "status": "reserved" if changed else "unknown",
            "changed": changed,
        }

    def mark_unknown_batch(
        self,
        reservation_ids: Sequence[str],
        *,
        expected_model: str,
        error: str,
    ) -> dict[str, Any]:
        """Atomically mark an exact outstanding reservation set unknown.

        This operator-recovery primitive never releases estimates. It is
        idempotent only when every supplied row is already unknown with the
        same timeout reason and the ledger has no other reserved rows.
        """
        with self._locked():
            data = self._read_unlocked()
            if data is None:
                raise BudgetError("cannot recover reservations in an empty ledger")
            self._check_binding(data)
            selected, changed = self._validate_unknown_batch(
                data, reservation_ids, expected_model=expected_model, error=error,
            )
            if not changed:
                return {
                    "reservation_ids": list(reservation_ids),
                    "status": "unknown",
                    "changed": False,
                }
            for row in selected:
                row["status"] = "unknown"
                row["actual_usd"] = None
                row["error"] = error
            self._write_unlocked(data)
            return {
                "reservation_ids": list(reservation_ids),
                "status": "unknown",
                "changed": True,
            }

    def summary(self) -> dict[str, Any]:
        """Return a read-only accounting summary."""
        with self._locked():
            data = self._read_unlocked() or self._new_data()
        rows = data["reservations"]
        imported_reserved = _money(self.carry_forward.get("imported_usd", 0.0), "imported_usd")
        imported_calibration = _money(
            self.carry_forward.get("imported_calibration_usd", 0.0),
            "imported_calibration_usd",
        )
        reserved = imported_reserved + sum(_reservation_charge(row) for row in rows)
        calibration_reserved = imported_calibration + sum(
            _reservation_charge(row)
            for row in rows if row.get("phase", "run") == "calibration"
        )
        local_estimated = sum(_money(row.get("estimated_usd", 0.0), "estimated_usd") for row in rows)
        actual_rows = [row for row in rows if row.get("actual_usd") is not None]
        measured_actual = sum(_money(row.get("actual_usd"), "actual_usd") for row in actual_rows)
        actual_complete = len(actual_rows) == len(rows) and not self.carry_forward
        return {
            "max_budget_usd": self.max_budget_usd,
            "accounting_basis": "local conservative estimate; not an invoice guarantee",
            "calibration_budget_usd": self.calibration_budget_usd,
            "estimated_usd": imported_reserved + local_estimated,
            "measured_actual_usd": measured_actual if not self.carry_forward else None,
            "actual_vs_estimate_usd": (measured_actual - (imported_reserved + local_estimated)) if actual_complete else None,
            "actual_usage_complete": actual_complete,
            "reserved_usd": reserved,
            "calibration_reserved_usd": calibration_reserved,
            "carried_forward_usd": imported_reserved,
            "carried_forward_calibration_usd": imported_calibration,
            "remaining_usd": max(0.0, self.max_budget_usd - reserved),
            "attempt_count": len(rows),
            "unknown_count": sum(row.get("status") == "unknown" for row in rows),
            "completed_count": sum(row.get("status") == "completed" for row in rows),
            "binding": dict(self.binding),
        }


__all__ = [
    "BUDGET_LEDGER_SCHEMA", "BudgetError", "BudgetExceeded", "BudgetLedger",
    "DEFAULT_CALIBRATION_BUDGET_USD", "DEFAULT_TOTAL_BUDGET_USD", "InvalidUsage",
    "TotalBudgetExceeded",
    "LedgerBindingError", "Pricing", "Reservation", "canonical_hash", "import_prior_ledger",
    "sha256_file", "validate_usage",
]


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#    No additional dependencies; use the repository environment.
# 2. Basic usage:
#    Import BudgetLedger from faithful_agent.py or faithful_s36.py.
# 3. Expected output:
#    An atomic JSON ledger with reservations and binding metadata.
#
# ═══════════════════════════════════════════════════════════════
