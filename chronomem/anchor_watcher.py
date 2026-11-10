"""M3 — code & file fingerprint anchor drift detector (PRD §3.3).

Memory atoms written for engineering contexts may bind an ``anchor_path``
pointing at a workspace file, optionally at a specific Python AST symbol
(``src/payment.py::RefundService.process``). At write time the SHA256
fingerprint of the file content (or of the symbol's source segment) is
persisted in ``anchor_sha256``.

:class:`AnchorWatcher.verify_anchors` re-scans the bound workspace files and
transitions atoms whose anchor has drifted to ``status='anchor_drifted'``:

* bound file deleted            -> ``file_deleted``
* bound AST symbol disappeared  -> ``symbol_missing``
* file/symbol content changed   -> ``content_changed``
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chronomem.store import Atom, MemoryStore

__all__ = ["AnchorWatcher", "AnchorVerdict", "parse_anchor",
           "compute_file_sha256", "compute_symbol_fingerprint"]

VERDICT_OK = "ok"
VERDICT_DRIFTED = "drifted"


def parse_anchor(anchor_path: str) -> Tuple[str, Optional[str]]:
    """Split ``file.py::Class.method`` into ``(file, qualname_or_None)``."""
    if "::" in anchor_path:
        path, qualname = anchor_path.split("::", 1)
        return path, qualname or None
    return anchor_path, None


def compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_ast_node(tree: ast.AST, qualname: str) -> Optional[ast.AST]:
    """Locate a module-level/nested node by dotted qualname."""
    node: ast.AST = tree
    for part in qualname.split("."):
        if not hasattr(node, "body"):
            return None
        child = None
        for stmt in node.body:  # type: ignore[attr-defined]
            if isinstance(
                stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and stmt.name == part:
                child = stmt
                break
        if child is None:
            return None
        node = child
    return node


def compute_symbol_fingerprint(path: Path, qualname: str) -> Optional[str]:
    """SHA256 over the source segment of the AST node named ``qualname``.

    Returns ``None`` when the file cannot be parsed or the symbol is absent.
    Formatting-sensitive by design: any edit inside the symbol changes the
    fingerprint, which is exactly the drift signal ChronoMem needs.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    node = _find_ast_node(tree, qualname)
    if node is None:
        return None
    segment = ast.get_source_segment(source, node)
    payload = segment if segment is not None else ast.dump(node)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_anchor_fingerprint(workspace_root: Path, anchor_path: str) -> Optional[str]:
    """Fingerprint for an anchor spec: file hash, or AST symbol segment hash."""
    rel, qualname = parse_anchor(anchor_path)
    path = workspace_root / rel
    if not path.is_file():
        return None
    if qualname is None:
        try:
            return compute_file_sha256(path)
        except OSError:
            return None
    return compute_symbol_fingerprint(path, qualname)


@dataclass(frozen=True)
class AnchorVerdict:
    atom_id: int
    anchor_path: str
    verdict: str  # "ok" | "drifted"
    reason: Optional[str] = None  # file_deleted | symbol_missing | content_changed
    current_sha256: Optional[str] = None


class AnchorWatcher:
    """Detects external code drift for anchor-bound memory atoms."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ------------------------------------------------------------------ #
    def bind(
        self,
        entity: str,
        slot: str,
        value,
        turn_id: int,
        anchor_path: str,
        workspace_root,
        *,
        namespace: str = "default",
        summary_text: str = "",
        source_turn_id: Optional[str] = None,
    ) -> Atom:
        """Write an anchor-bound memory atom, persisting the fingerprint."""
        fp = compute_anchor_fingerprint(Path(workspace_root), anchor_path)
        return self.store.insert_atom(
            entity, slot, value, turn_id,
            namespace=namespace,
            summary_text=summary_text,
            source_turn_id=source_turn_id,
            anchor_path=anchor_path,
            anchor_sha256=fp,
        )

    # ------------------------------------------------------------------ #
    def verify_anchors(
        self,
        workspace_root,
        *,
        namespace: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> List[AnchorVerdict]:
        """Scan all active anchor-bound atoms and mark drifted ones.

        Returns one :class:`AnchorVerdict` per checked atom. Drifted atoms are
        transitioned to ``status='anchor_drifted'`` (single-active invariant
        preserved; ``valid_to_turn`` closed at ``turn_id`` or the store's max
        turn).
        """
        root = Path(workspace_root)
        if turn_id is None:
            turn_id = self.store.max_turn()

        verdicts: List[AnchorVerdict] = []
        for atom in self.store.all_atoms(namespace=namespace):
            if atom.status != "active" or not atom.anchor_path:
                continue
            verdict = self._check_atom(root, atom)
            verdicts.append(verdict)
            if verdict.verdict == VERDICT_DRIFTED:
                self.store.deactivate(
                    atom.atom_id,
                    new_status="anchor_drifted",
                    valid_to_turn=turn_id,
                )
        return verdicts

    # ------------------------------------------------------------------ #
    def _check_atom(self, root: Path, atom: Atom) -> AnchorVerdict:
        rel, qualname = parse_anchor(atom.anchor_path)  # type: ignore[arg-type]
        path = root / rel
        current: Optional[str]

        if not path.is_file():
            current = None
            reason = "file_deleted"
        elif qualname is None:
            try:
                current = compute_file_sha256(path)
            except OSError:
                current = None
            reason = None if current == atom.anchor_sha256 else "content_changed"
        else:
            current = compute_symbol_fingerprint(path, qualname)
            if current is None:
                reason = "symbol_missing"
            elif current != atom.anchor_sha256:
                reason = "content_changed"
            else:
                reason = None

        if reason is None and current is not None:
            return AnchorVerdict(atom.atom_id, atom.anchor_path, VERDICT_OK,
                                 None, current)
        return AnchorVerdict(atom.atom_id, atom.anchor_path, VERDICT_DRIFTED,
                             reason or "content_changed", current)

    # ------------------------------------------------------------------ #
    def drift_reasons(self, workspace_root, *, namespace=None) -> Dict[int, str]:
        """Convenience map ``atom_id -> drift reason`` for drifted atoms."""
        return {
            v.atom_id: v.reason
            for v in self.verify_anchors(workspace_root, namespace=namespace)
            if v.verdict == VERDICT_DRIFTED
        }
