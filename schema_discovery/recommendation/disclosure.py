from __future__ import annotations

from .schema import NormalisationDisclosure


def disclose(applied: list[str] | None = None, notes: str = "") -> NormalisationDisclosure:
    return NormalisationDisclosure(applied=list(applied or []), notes=notes)


def disclose_key_normalisation(*, trimmed: bool = False, casefolded: bool = False, notes: str = "") -> NormalisationDisclosure:
    applied: list[str] = []
    if trimmed:
        applied.append("trim_whitespace")
    if casefolded:
        applied.append("casefold_strings")
    return NormalisationDisclosure(applied=applied, notes=notes)


def disclose_numeric_coercion(notes: str = "") -> NormalisationDisclosure:
    return NormalisationDisclosure(applied=["numeric_coercion"], notes=notes)
