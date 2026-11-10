"""M2 — write-path 4-state conflict resolution state machine (PRD §3.2).

When a new conversation turn ``Turn_t`` produces a candidate state change,
:class:`ConflictResolver` compares it against the current ``active`` record
of ``(namespace, entity, slot)`` and performs exactly one deterministic
state transition:

``ADD``
    no active record exists -> insert new active atom.
``SUPERSEDE``
    active exists and the new value is a mutually-exclusive overwrite ->
    atomic version-chain update (old -> stale, new -> active).
``MERGE_PARTIAL``
    the slot holds a composite dict and the input only modifies
    sub-attributes -> deep-merge into a complete new snapshot, then walk
    the SUPERSEDE version chain so history stays traceable.
``RETRACT``
    the user explicitly cancels a setting -> mark the active atom
    ``retracted`` so retrieval can abstain later.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

from chronomem.store import Atom, MemoryStore

__all__ = ["ConflictResolver", "Resolution", "ResolutionResult", "deep_merge"]


class Resolution(str, enum.Enum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    MERGE_PARTIAL = "MERGE_PARTIAL"
    RETRACT = "RETRACT"


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of one write through the conflict-resolution state machine."""

    operation: Resolution
    namespace: str
    entity: str
    slot: str
    old_atom: Optional[Atom]
    new_atom: Optional[Atom]  # None for RETRACT
    merged_value: Any = None  # set for MERGE_PARTIAL (the merged snapshot)


def deep_merge(old: Any, patch: Any) -> Any:
    """Recursive incremental merge of ``patch`` into ``old``.

    * dict ∩ dict -> recursive merge (patch wins on leaf conflicts);
    * anything else (lists included) -> patch replaces wholesale.
    """
    if isinstance(old, dict) and isinstance(patch, dict):
        out: Dict[str, Any] = dict(old)
        for key, value in patch.items():
            if key in out and isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return patch


class ConflictResolver:
    """Applies the 4-state transition table on top of :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore, namespace: str = "default") -> None:
        self.store = store
        self.namespace = namespace

    # ------------------------------------------------------------------ #
    def write(
        self,
        entity: str,
        slot: str,
        value: Any,
        turn_id: int,
        *,
        summary_text: str = "",
        source_turn_id: Optional[str] = None,
        anchor_path: Optional[str] = None,
        anchor_sha256: Optional[str] = None,
        force_merge: bool = False,
    ) -> ResolutionResult:
        """Resolve a candidate state change into one of ADD/SUPERSEDE/MERGE_PARTIAL.

        ``force_merge=True`` treats a dict input whose keys extend the old
        schema as a partial update instead of a schema replacement.
        """
        active = self.store.get_active(entity, slot, namespace=self.namespace)

        if active is None:
            atom = self.store.insert_atom(
                entity,
                slot,
                value,
                turn_id,
                namespace=self.namespace,
                summary_text=summary_text,
                source_turn_id=source_turn_id,
                anchor_path=anchor_path,
                anchor_sha256=anchor_sha256,
            )
            return ResolutionResult(Resolution.ADD, self.namespace, entity, slot,
                                    None, atom)

        old_value = active.value
        is_partial = (
            isinstance(old_value, dict)
            and isinstance(value, dict)
            and set(value.keys()) <= set(old_value.keys())
        ) or (force_merge and isinstance(old_value, dict) and isinstance(value, dict))

        if is_partial:
            merged = deep_merge(old_value, value)
            _, new_atom = self.store.supersede(
                entity,
                slot,
                merged,
                turn_id,
                namespace=self.namespace,
                summary_text=summary_text,
                source_turn_id=source_turn_id,
                anchor_path=anchor_path,
                anchor_sha256=anchor_sha256,
            )
            # re-fetch: `active` was materialized before the transaction
            old_atom = self.store.get_atom(active.atom_id)
            return ResolutionResult(Resolution.MERGE_PARTIAL, self.namespace,
                                    entity, slot, old_atom, new_atom, merged)

        _, new_atom = self.store.supersede(
            entity,
            slot,
            value,
            turn_id,
            namespace=self.namespace,
            summary_text=summary_text,
            source_turn_id=source_turn_id,
            anchor_path=anchor_path,
            anchor_sha256=anchor_sha256,
        )
        old_atom = self.store.get_atom(active.atom_id)
        return ResolutionResult(Resolution.SUPERSEDE, self.namespace, entity,
                                slot, old_atom, new_atom)

    # ------------------------------------------------------------------ #
    def retract(
        self,
        entity: str,
        slot: str,
        turn_id: int,
        *,
        source_turn_id: Optional[str] = None,
    ) -> ResolutionResult:
        """Explicit cancellation (RETRACT transition)."""
        active = self.store.get_active(entity, slot, namespace=self.namespace)
        if active is None:
            # nothing to retract — deterministic no-op
            return ResolutionResult(Resolution.RETRACT, self.namespace, entity,
                                    slot, None, None)
        updated = self.store.deactivate(
            active.atom_id, new_status="retracted", valid_to_turn=turn_id
        )
        return ResolutionResult(Resolution.RETRACT, self.namespace, entity, slot,
                                updated, None)
