"""Typed pass pipelines.

Pipelines are built from ``Pass`` objects (generated per MLIR pass in
``mlir_python.passes``) and ``Nested`` groups anchored on operation classes.
They render MLIR's textual pipeline syntax internally; ``PassManager.parse``
accepts that syntax directly as a fallback.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import ClassVar

from ._mlir_python import Context, Module, Operation, ParsedPassPipeline

__all__ = ["Nested", "Pass", "PassManager", "PipelineElement"]


def _operation_name(anchor: type[Operation]) -> str:
    """The name MLIR uses for pipelines anchored on ``anchor``."""
    if anchor is Operation:
        return "any"
    name = getattr(anchor, "OPERATION_NAME", None)
    if not isinstance(name, str):
        raise TypeError(
            f"{anchor.__name__} is not an operation class with a known name"
        )
    return name


def _format_option(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, (Nested, Pass)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return "{" + ",".join(_format_option(item) for item in value) + "}"
    text = str(value)
    if text == "" or any(c in text for c in " ,{}=\"'"):
        return "{" + text + "}"
    return text


@dataclasses.dataclass(frozen=True, kw_only=True)
class Pass:
    """Base class of the generated pass classes in ``mlir_python.passes``.

    Each subclass is one MLIR pass; its dataclass fields are the pass's
    options, defaulting to MLIR's defaults.
    """

    ARGUMENT: ClassVar[str]
    """The pass's name in MLIR's textual pipeline syntax."""
    ANCHOR: ClassVar[type[Operation] | None] = None
    """The operation class the pass must run on, or ``None`` for any."""
    ANCHOR_INTERFACE: ClassVar[str | None] = None
    """For interface passes, the interface the operation must implement."""

    def __post_init__(self) -> None:
        # Sequences are stored as tuples so passes stay hashable and immutable.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, list):
                object.__setattr__(self, field.name, tuple(value))

    def __str__(self) -> str:
        options = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None or value == field.default:
                continue
            options.append(f"{field.metadata['argument']}={_format_option(value)}")
        return self.ARGUMENT + ("{" + " ".join(options) + "}" if options else "")


type PipelineElement = Pass | Nested
"""What a pipeline contains: passes, and groups nested on inner operations."""


def _check_anchor(element: PipelineElement, anchor: type[Operation]) -> None:
    if isinstance(element, Pass):
        required = element.ANCHOR
        if (
            required is not None
            and anchor is not Operation
            and not issubclass(anchor, required)
        ):
            raise ValueError(
                f"{type(element).__name__} runs on {required.__name__}, "
                f"not {anchor.__name__}; wrap it in Nested({required.__name__}, [...])"
            )
    elif not isinstance(element, Nested):
        raise TypeError(f"expected a Pass or Nested, got {type(element).__name__}")


class Nested:
    """Runs ``passes`` on every ``anchor`` operation nested directly inside
    the operation the enclosing pipeline runs on.

    Example: ``Nested(func.FuncOp, [passes.CSE()])`` inside a pipeline on
    ``Module`` runs CSE on each function.
    """

    def __init__(
        self, anchor: type[Operation], passes: Sequence[PipelineElement]
    ) -> None:
        self.anchor = anchor
        self.passes: tuple[PipelineElement, ...] = tuple(passes)
        for element in self.passes:
            _check_anchor(element, anchor)

    def __str__(self) -> str:
        return (
            f"{_operation_name(self.anchor)}({','.join(str(p) for p in self.passes)})"
        )

    def __repr__(self) -> str:
        return f"Nested({self.anchor.__name__}, {list(self.passes)!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Nested)
            and other.anchor is self.anchor
            and other.passes == self.passes
        )

    def __hash__(self) -> int:
        return hash((self.anchor, self.passes))


class PassManager:
    """Runs a pipeline of passes on operations of one class.

    Example::

        pm = PassManager(Module, [passes.Canonicalizer(), Nested(func.FuncOp, [passes.CSE()])])
        pm.run(module)
    """

    def __init__(
        self,
        anchor: type[Operation] = Module,
        passes: Sequence[PipelineElement] = (),
        *,
        verify_each: bool = True,
    ) -> None:
        """Create a pass manager.

        Args:
            anchor: The operation class ``run`` accepts; ``Operation`` accepts
                any operation (and then only op-agnostic passes fit directly).
            passes: The pipeline, in order.
            verify_each: Verify the IR after every pass.

        Raises:
            ValueError: If a pass cannot run on the operation it is placed on.
        """
        self.anchor = anchor
        self.verify_each = verify_each
        self._passes: list[PipelineElement] = []
        self._parsed: dict[Context, ParsedPassPipeline] = {}
        self.add(*passes)

    @property
    def passes(self) -> tuple[PipelineElement, ...]:
        """The pipeline, in order."""
        return tuple(self._passes)

    def add(self, *passes: PipelineElement) -> None:
        """Append passes to the pipeline.

        Raises:
            ValueError: If a pass cannot run on this manager's anchor.
        """
        for element in passes:
            _check_anchor(element, self.anchor)
        self._passes.extend(passes)
        self._parsed.clear()

    def run(self, op: Operation) -> None:
        """Run the pipeline on ``op`` in place.

        Raises:
            TypeError: If ``op`` is not an instance of the anchor class.
            MLIRError: If a pass fails or the IR does not verify afterwards.
        """
        if not isinstance(op, self.anchor):
            raise TypeError(
                f"this pass manager runs on {self.anchor.__name__}, "
                f"not {type(op).__name__}"
            )
        parsed = self._parsed.get(op.context)
        if parsed is None:
            parsed = ParsedPassPipeline.parse(str(self), context=op.context)
            parsed.enable_verifier(self.verify_each)
            self._parsed[op.context] = parsed
        parsed.run(op)

    @staticmethod
    def parse(pipeline: str, *, context: Context | None = None) -> ParsedPassPipeline:
        """Fallback: build a pipeline from MLIR's textual syntax, e.g.
        ``"builtin.module(canonicalize, func.func(cse))"``.

        Raises:
            ValueError: If the text does not parse.
        """
        return ParsedPassPipeline.parse(pipeline, context=context)

    def __str__(self) -> str:
        """The pipeline in MLIR's textual syntax."""
        return (
            f"{_operation_name(self.anchor)}({','.join(str(p) for p in self._passes)})"
        )

    def __repr__(self) -> str:
        return f"PassManager({self.anchor.__name__}, {self._passes!r})"
