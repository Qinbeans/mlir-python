"""Scalar types for ``mlir_python.lang`` function signatures.

Annotate parameters, results, and variables with these to fix their machine
representation; Python's own ``int``, ``float``, and ``bool`` mean ``i64``,
``f64``, and a 1-bit boolean.

Type checkers see ``i32`` and friends as aliases of ``int`` (``f32``/``f64``
of ``float``, ``cstr`` of ``str``; see ``types.pyi``), so function bodies are
ordinary, fully type-checked Python. Calling a type converts: ``i32(x)``,
``f64(n)``. At runtime they are ``ScalarType`` objects (see ``_types.py``).
"""

from ._types import (
    Array as Array,
)
from ._types import (
    ArrayType as ArrayType,
)
from ._types import (
    Fn as Fn,
)
from ._types import (
    FnType as FnType,
)
from ._types import (
    Ptr as Ptr,
)
from ._types import (
    ScalarType as ScalarType,
)
from ._types import (
    StructType as StructType,
)
from ._types import (
    array as array,
)
from ._types import (
    cstr as cstr,
)
from ._types import (
    f32 as f32,
)
from ._types import (
    f64 as f64,
)
from ._types import (
    i8 as i8,
)
from ._types import (
    i16 as i16,
)
from ._types import (
    i32 as i32,
)
from ._types import (
    i64 as i64,
)
from ._types import (
    ptr as ptr,
)
from ._types import (
    stack as stack,
)
from ._types import (
    struct as struct,
)
from ._types import (
    u8 as u8,
)
from ._types import (
    u16 as u16,
)
from ._types import (
    u32 as u32,
)
from ._types import (
    u64 as u64,
)

__all__ = [
    "Array",
    "ArrayType",
    "Fn",
    "FnType",
    "Ptr",
    "ScalarType",
    "StructType",
    "array",
    "cstr",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "ptr",
    "stack",
    "struct",
    "u8",
    "u16",
    "u32",
    "u64",
]
