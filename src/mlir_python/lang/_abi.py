"""The C calling convention for calls to ``@extern`` functions.

MLIR passes a struct value as its separate fields, which is not how C passes
it. C compilers classify each struct by size and field types and rewrite it
into registers or memory, differently on each platform; this module makes the
same decisions as Clang for the two supported ones:

- x86-64 (System V): a struct of up to 16 bytes travels as one or two 8-byte
  pieces, each in a general-purpose register (any integer field) or an SSE
  register (only float fields); larger structs, and structs that no longer
  fit in the remaining registers, are copied to the stack (``byval``).
- AArch64 (AAPCS64): 1-4 fields of one float type (an HFA) travel in float
  registers; other structs of up to 16 bytes in one or two general-purpose
  registers; larger ones as a pointer to a copy.

Larger results are returned through a pointer the caller passes (``sret``).
``bool`` and 8/16-bit integers are zero- or sign-extended, as C expects.

The rewriting itself goes through memory: a struct is stored to a stack slot
and reloaded as its pieces (and the reverse for results), which LLVM's
optimizer turns into register moves.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Literal

from .. import _mlir_python as ir
from ..dialects import llvm
from ._types import ScalarType, StructType, f32, f64

type Passing = Literal["direct", "pieces", "byval", "pointer"]
"""How one argument or result crosses the call:

- ``direct``: as its own value (scalars);
- ``pieces``: a struct as the values at ``Lowered.pieces`` byte offsets;
- ``byval``: a struct as a pointer to a copy the callee owns (x86-64 stack
  copy) or, for a result, a pointer the callee writes (``sret``);
- ``pointer``: a struct as a pointer to a caller-made copy (AArch64).
"""


@dataclass
class Lowered:
    """One parameter or the result, as passed at the machine level."""

    kind: ScalarType
    passing: Passing
    pieces: list[tuple[int, ir.Type]] = field(default_factory=list)


@dataclass
class ExternABI:
    """An extern's declaration as C compilers see it."""

    params: list[Lowered]
    result: Lowered | None
    function_type: llvm.FunctionType
    arg_attrs: ir.ArrayAttr
    res_attrs: ir.ArrayAttr | None


class UnsupportedABI(Exception):
    """A struct crosses a call on a platform whose C ABI is not implemented."""


def _target() -> Literal["x86_64", "aarch64"] | None:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return None


def _scalars(kind: ScalarType, base: int = 0) -> list[tuple[int, ScalarType]]:
    """Every scalar field of ``kind`` with its byte offset, nested structs
    flattened."""
    if not isinstance(kind, StructType):
        return [(base, kind)]
    flat: list[tuple[int, ScalarType]] = []
    for offset, (_, field_kind) in zip(kind.offsets(), kind.fields, strict=True):
        flat.extend(_scalars(field_kind, base + offset))
    return flat


def _i64() -> ir.Type:
    return ir.IntegerType(64)


# -- x86-64 System V -----------------------------------------------------------


def _x86_eightbytes(kind: StructType) -> list[ir.Type] | None:
    """The register piece for each 8 bytes of a struct of up to 16 bytes, or
    ``None`` when it is passed in memory."""
    if kind.size > 16:
        return None
    flat = _scalars(kind)
    pieces: list[ir.Type] = []
    for start in range(0, kind.size, 8):
        members = [k for offset, k in flat if start <= offset < start + 8]
        if members and all(k.kind == "float" for k in members):
            if any(k == f64 for k in members):
                pieces.append(f64.mlir())
            elif any(offset == start + 4 for offset, k in flat if k == f32):
                pieces.append(ir.VectorType([2], f32.mlir()))
            else:
                pieces.append(f32.mlir())
        else:
            pieces.append(_i64())
    return pieces


def _is_sse(piece: ir.Type) -> bool:
    return not isinstance(piece, ir.IntegerType)


def _x86(params: list[ScalarType], result: ScalarType | None) -> list[Lowered | None]:
    """Lowered params, then the result (``None`` for no result)."""
    free_int, free_sse = 6, 8
    lowered_result: Lowered | None = None
    if result is not None:
        if isinstance(result, StructType):
            pieces = _x86_eightbytes(result)
            if pieces is None:
                lowered_result = Lowered(result, "byval")
                free_int -= 1  # the sret pointer
            else:
                lowered_result = Lowered(
                    result, "pieces", [(8 * i, p) for i, p in enumerate(pieces)]
                )
        else:
            lowered_result = Lowered(result, "direct")
    lowered: list[Lowered | None] = []
    for kind in params:
        if not isinstance(kind, StructType):
            if kind.kind == "float":
                free_sse -= 1
            else:
                free_int -= 1
            lowered.append(Lowered(kind, "direct"))
            continue
        pieces = _x86_eightbytes(kind)
        if pieces is not None:
            sse = sum(_is_sse(p) for p in pieces)
            ints = len(pieces) - sse
            if ints <= free_int and sse <= free_sse:
                free_int -= ints
                free_sse -= sse
                lowered.append(
                    Lowered(kind, "pieces", [(8 * i, p) for i, p in enumerate(pieces)])
                )
                continue
        lowered.append(Lowered(kind, "byval"))
    return [*lowered, lowered_result]


# -- AArch64 AAPCS64 -----------------------------------------------------------


def _aarch64_pieces(kind: StructType) -> list[tuple[int, ir.Type]] | None:
    """A struct's register pieces, or ``None`` when it is passed through
    memory."""
    flat = _scalars(kind)
    first = flat[0][1]
    if (
        len(flat) <= 4
        and first.kind == "float"
        and all(k == first for _, k in flat)
        and kind.size == len(flat) * first.size
    ):
        return [(0, llvm.ArrayType(first.mlir(), len(flat)))]  # an HFA
    if kind.size <= 8:
        return [(0, _i64())]
    if kind.size <= 16:
        return [(0, llvm.ArrayType(_i64(), 2))]
    return None


def _aarch64(
    params: list[ScalarType], result: ScalarType | None
) -> list[Lowered | None]:
    def lower(kind: ScalarType, memory: Passing) -> Lowered:
        if not isinstance(kind, StructType):
            return Lowered(kind, "direct")
        pieces = _aarch64_pieces(kind)
        return (
            Lowered(kind, memory) if pieces is None else Lowered(kind, "pieces", pieces)
        )

    lowered: list[Lowered | None] = [lower(kind, "pointer") for kind in params]
    return [*lowered, None if result is None else lower(result, "byval")]


# -- declarations ----------------------------------------------------------------


def _extension(kind: ScalarType) -> dict[str, ir.Attribute]:
    """C passes ``bool`` and small integers widened to a full register."""
    if kind.kind == "bool" or (kind.kind == "uint" and kind.bits < 32):
        return {"llvm.zeroext": ir.UnitAttr()}
    if kind.kind == "int" and kind.bits < 32:
        return {"llvm.signext": ir.UnitAttr()}
    return {}


def lower(
    params: list[ScalarType], result: ScalarType | None, *, variadic: bool
) -> ExternABI:
    """The C-level declaration of an extern taking ``params`` and returning
    ``result`` (``None`` for ``void``). Needs an active context.

    Raises:
        UnsupportedABI: If a struct is passed or returned on a platform other
            than x86-64 or AArch64.
    """
    structs = [k for k in [*params, result] if isinstance(k, StructType)]
    target = _target()
    if structs and target is None:
        raise UnsupportedABI(
            f"passing {structs[0].name} by value to C is not implemented on "
            f"{platform.machine()}; pass a Ptr[{structs[0].name}] instead"
        )
    if target == "aarch64":
        *lowered_params, lowered_result = _aarch64(params, result)
    else:
        *lowered_params, lowered_result = _x86(params, result)
    pointer = llvm.PointerType()
    inputs: list[ir.Type] = []
    attrs: list[ir.DictAttr] = []
    output: ir.Type = llvm.VoidType()
    res_attrs: ir.ArrayAttr | None = None
    if lowered_result is not None:
        kind = lowered_result.kind
        if lowered_result.passing == "byval":
            inputs.append(pointer)
            attrs.append(ir.DictAttr({"llvm.sret": ir.TypeAttr(kind.mlir())}))
        elif lowered_result.passing == "pieces":
            types = [t for _, t in lowered_result.pieces]
            output = types[0] if len(types) == 1 else llvm.StructType(types)
        else:
            output = kind.c_type()
            if extension := _extension(kind):
                res_attrs = ir.ArrayAttr([ir.DictAttr(extension)])
    for item in lowered_params:
        assert item is not None
        if item.passing == "direct":
            inputs.append(item.kind.c_type())
            attrs.append(ir.DictAttr(_extension(item.kind)))
        elif item.passing == "pieces":
            for _, piece in item.pieces:
                inputs.append(piece)
                attrs.append(ir.DictAttr({}))
        elif item.passing == "byval":
            inputs.append(pointer)
            alignment = max(8, item.kind.alignment)
            attrs.append(
                ir.DictAttr(
                    {
                        "llvm.byval": ir.TypeAttr(item.kind.mlir()),
                        "llvm.align": ir.IntegerAttr(alignment, ir.IntegerType(64)),
                    }
                )
            )
        else:
            inputs.append(pointer)
            attrs.append(ir.DictAttr({}))
    return ExternABI(
        [p for p in lowered_params if p is not None],
        lowered_result,
        llvm.FunctionType(output, inputs, variadic=variadic),
        ir.ArrayAttr(attrs, context=ir.Context.current()),
        res_attrs,
    )
