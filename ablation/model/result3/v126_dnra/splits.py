"""Leakage guards for outer-cohort and participant-grouped splits."""
from __future__ import annotations


class SplitLeakageError(ValueError):
    """Raised when a split would expose held-out data during development."""


def validate_outer_split(heldout_cohort: str, development_cohorts: set[str]) -> None:
    if heldout_cohort in development_cohorts:
        raise SplitLeakageError("HELDOUT_COHORT_IN_DEVELOPMENT")
    if not development_cohorts:
        raise SplitLeakageError("EMPTY_DEVELOPMENT_COHORT_SET")


def validate_anchor_split(train_participants: list[str], validation_participants: list[str]) -> None:
    overlap = set(map(str, train_participants)).intersection(map(str, validation_participants))
    if overlap:
        raise SplitLeakageError(f"ANCHOR_PARTICIPANT_OVERLAP:{sorted(overlap)}")
    if not train_participants or not validation_participants:
        raise SplitLeakageError("EMPTY_ANCHOR_PARTITION")
