"""Typed operations for each linked MLIR dialect.

Import a dialect module to build its operations with checked, named
parameters, for example ``from mlir_python.dialects import arith`` and then
``arith.AddIOp(lhs, rhs)``. Every class is an ``Operation`` subclass, and IR
navigation returns these classes, so ``isinstance`` narrows operations.

The modules are generated from MLIR's operation definitions by
``tools/gen_dialects.py``.
"""
