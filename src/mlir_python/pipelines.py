"""Typed versions of MLIR's named pass pipelines, as lists of passes to run
with ``PassManager`` or extend::

    PassManager(Module, [*pipelines.bufferize(), *codegen.llvm_lowering_pipeline()])

``bufferize`` turns value-semantic tensors into buffers (memrefs);
``buffer_deallocation`` frees every buffer once nothing uses it.
"""

from __future__ import annotations

from . import passes
from ._passes import PipelineElement

__all__ = ["buffer_deallocation", "bufferize"]


def buffer_deallocation(
    *, private_function_dynamic_ownership: bool = False
) -> list[PipelineElement]:
    """Free every heap buffer (``memref.alloc``) once it is dead, like MLIR's
    ``buffer-deallocation-pipeline``.

    Ownership is tracked through branches, loops, and calls: a function frees
    the buffers it allocates unless it returns them, and a caller owns the
    buffers a call returns. Buffers passed as arguments are borrowed.

    Args:
        private_function_dynamic_ownership: Let private functions receive
            ownership of their buffer arguments at runtime, which can save
            copies at the cost of an extra flag per argument.
    """
    return [
        passes.ExpandRealloc(emit_deallocs=False),
        passes.Canonicalizer(),
        passes.OwnershipBasedBufferDeallocation(
            private_function_dynamic_ownership=private_function_dynamic_ownership
        ),
        passes.Canonicalizer(),
        passes.BufferDeallocationSimplification(),
        passes.LowerDeallocations(),
        passes.CSE(),
        passes.Canonicalizer(),
    ]


def bufferize(*, deallocate: bool = True) -> list[PipelineElement]:
    """Turn tensors into buffers with one-shot bufferization, across function
    boundaries (tensor arguments and results become memrefs with the identity
    layout), then free the buffers (see ``buffer_deallocation``).

    Args:
        deallocate: Also run ``buffer_deallocation``.
    """
    return [
        passes.OneShotBufferize(
            bufferize_function_boundaries=True,
            function_boundary_type_conversion=passes.LayoutMapOption.IDENTITY_LAYOUT_MAP,
        ),
        *(buffer_deallocation() if deallocate else []),
    ]
