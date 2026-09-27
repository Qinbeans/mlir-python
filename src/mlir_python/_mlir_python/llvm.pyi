"""Typed operations of the MLIR ``llvm`` dialect."""

import enum
from collections.abc import Sequence

import mlir_python._mlir_python

class IntegerOverflowFlags(enum.Flag):
    """LLVM integer overflow flags"""

    NONE = 0
    """``none`` in MLIR text."""

    NSW = 1
    """``nsw`` in MLIR text."""

    NUW = 2
    """``nuw`` in MLIR text."""

class Linkage(enum.Enum):
    """LLVM linkage types"""

    EXTERNAL = 0
    """``external`` in MLIR text."""

    AVAILABLE_EXTERNALLY = 1
    """``available_externally`` in MLIR text."""

    LINKONCE = 2
    """``linkonce`` in MLIR text."""

    LINKONCE_ODR = 3
    """``linkonce_odr`` in MLIR text."""

    WEAK = 4
    """``weak`` in MLIR text."""

    WEAK_ODR = 5
    """``weak_odr`` in MLIR text."""

    APPENDING = 6
    """``appending`` in MLIR text."""

    INTERNAL = 7
    """``internal`` in MLIR text."""

    PRIVATE = 8
    """``private`` in MLIR text."""

    EXTERN_WEAK = 9
    """``extern_weak`` in MLIR text."""

    COMMON = 10
    """``common`` in MLIR text."""

class UnnamedAddr(enum.Enum):
    """LLVM GlobalValue UnnamedAddr"""

    NONE = 0
    """NONE"""

    LOCAL = 1
    """``local_unnamed_addr`` in MLIR text."""

    GLOBAL = 2
    """``unnamed_addr`` in MLIR text."""

class Visibility(enum.Enum):
    """LLVM GlobalValue Visibility"""

    DEFAULT = 0
    """DEFAULT"""

    HIDDEN = 1
    """``hidden`` in MLIR text."""

    PROTECTED = 2
    """``protected`` in MLIR text."""

class AtomicOrdering(enum.Enum):
    """Atomic ordering for LLVM's memory model"""

    NOT_ATOMIC = 0
    """``not_atomic`` in MLIR text."""

    UNORDERED = 1
    """``unordered`` in MLIR text."""

    MONOTONIC = 2
    """``monotonic`` in MLIR text."""

    ACQUIRE = 4
    """``acquire`` in MLIR text."""

    RELEASE = 5
    """``release`` in MLIR text."""

    ACQ_REL = 6
    """``acq_rel`` in MLIR text."""

    SEQ_CST = 7
    """``seq_cst`` in MLIR text."""

class AtomicBinOp(enum.Enum):
    """llvm.atomicrmw binary operations"""

    XCHG = 0
    """``xchg`` in MLIR text."""

    ADD = 1
    """``add`` in MLIR text."""

    SUB = 2
    """``sub`` in MLIR text."""

    NAND = 4
    """``nand`` in MLIR text."""

    MAX = 7
    """``max`` in MLIR text."""

    MIN = 8
    """``min`` in MLIR text."""

    UMAX = 9
    """``umax`` in MLIR text."""

    UMIN = 10
    """``umin`` in MLIR text."""

    FADD = 11
    """``fadd`` in MLIR text."""

    FSUB = 12
    """``fsub`` in MLIR text."""

    FMAX = 13
    """``fmax`` in MLIR text."""

    FMIN = 14
    """``fmin`` in MLIR text."""

    UINC_WRAP = 15
    """``uinc_wrap`` in MLIR text."""

    UDEC_WRAP = 16
    """``udec_wrap`` in MLIR text."""

    USUB_COND = 17
    """``usub_cond`` in MLIR text."""

    USUB_SAT = 18
    """``usub_sat`` in MLIR text."""

    FMAXIMUM = 19
    """``fmaximum`` in MLIR text."""

    FMINIMUM = 20
    """``fminimum`` in MLIR text."""

class FastmathFlags(enum.Flag):
    """LLVM fastmath flags"""

    NONE = 0
    """``none`` in MLIR text."""

    NNAN = 1
    """``nnan`` in MLIR text."""

    NINF = 2
    """``ninf`` in MLIR text."""

    NSZ = 4
    """``nsz`` in MLIR text."""

    ARCP = 8
    """``arcp`` in MLIR text."""

    CONTRACT = 16
    """``contract`` in MLIR text."""

    AFN = 32
    """``afn`` in MLIR text."""

    REASSOC = 64
    """``reassoc`` in MLIR text."""

    FAST = 127
    """``fast`` in MLIR text."""

class Comdat(enum.Enum):
    """LLVM Comdat Types"""

    ANY = 0
    """``any`` in MLIR text."""

    EXACT_MATCH = 1
    """``exactmatch`` in MLIR text."""

    LARGEST = 2
    """``largest`` in MLIR text."""

    NO_DEDUPLICATE = 3
    """``nodeduplicate`` in MLIR text."""

    SAME_SIZE = 4
    """``samesize`` in MLIR text."""

class FCmpPredicate(enum.Enum):
    """llvm.fcmp comparison predicate"""

    OEQ = 1
    """``oeq`` in MLIR text."""

    OGT = 2
    """``ogt`` in MLIR text."""

    OGE = 3
    """``oge`` in MLIR text."""

    OLT = 4
    """``olt`` in MLIR text."""

    OLE = 5
    """``ole`` in MLIR text."""

    ONE = 6
    """``one`` in MLIR text."""

    ORD = 7
    """``ord`` in MLIR text."""

    UEQ = 8
    """``ueq`` in MLIR text."""

    UGT = 9
    """``ugt`` in MLIR text."""

    UGE = 10
    """``uge`` in MLIR text."""

    ULT = 11
    """``ult`` in MLIR text."""

    ULE = 12
    """``ule`` in MLIR text."""

    UNE = 13
    """``une`` in MLIR text."""

    UNO = 14
    """``uno`` in MLIR text."""

class GEPNoWrapFlags(enum.Flag):
    """::mlir::LLVM::GEPNoWrapFlags"""

    NONE = 0
    """``none`` in MLIR text."""

    INBOUNDS_FLAG = 1
    """``inbounds_flag`` in MLIR text."""

    NUSW = 2
    """``nusw`` in MLIR text."""

    NUW = 4
    """``nuw`` in MLIR text."""

    INBOUNDS = 3
    """``inbounds`` in MLIR text."""

class ICmpPredicate(enum.Enum):
    """llvm.icmp comparison predicate"""

    EQ = 0
    """``eq`` in MLIR text."""

    NE = 1
    """``ne`` in MLIR text."""

    SLT = 2
    """``slt`` in MLIR text."""

    SLE = 3
    """``sle`` in MLIR text."""

    SGT = 4
    """``sgt`` in MLIR text."""

    SGE = 5
    """``sge`` in MLIR text."""

    ULT = 6
    """``ult`` in MLIR text."""

    ULE = 7
    """``ule`` in MLIR text."""

    UGT = 8
    """``ugt`` in MLIR text."""

    UGE = 9
    """``uge`` in MLIR text."""

class AsmDialect(enum.Enum):
    """ATT (0) or Intel (1) asm dialect"""

    AD_ATT = 0
    """``att`` in MLIR text."""

    AD_INTEL = 1
    """``intel`` in MLIR text."""

class FramePointerKind(enum.Enum):
    """LLVM FramePointerKind"""

    NONE = 0
    """``none`` in MLIR text."""

    NON_LEAF = 1
    """``non-leaf`` in MLIR text."""

    ALL = 2
    """``all`` in MLIR text."""

    RESERVED = 3
    """``reserved`` in MLIR text."""

    NON_LEAF_NO_RESERVE = 4
    """``non-leaf-no-reserve`` in MLIR text."""

class UWTableKind(enum.Enum):
    """LLVM Unwind Behavior"""

    NONE = 0
    """``none`` in MLIR text."""

    SYNC = 1
    """``sync`` in MLIR text."""

    ASYNC = 2
    """``async`` in MLIR text."""

class DIFlags(enum.Flag):
    """LLVM DI flags"""

    ZERO = 0
    """``Zero`` in MLIR text."""

    BIT0 = 1
    """``Bit0`` in MLIR text."""

    BIT1 = 2
    """``Bit1`` in MLIR text."""

    PRIVATE = 1
    """``Private`` in MLIR text."""

    PROTECTED = 2
    """``Protected`` in MLIR text."""

    PUBLIC = 3
    """``Public`` in MLIR text."""

    FWD_DECL = 4
    """``FwdDecl`` in MLIR text."""

    APPLE_BLOCK = 8
    """``AppleBlock`` in MLIR text."""

    RESERVED_BIT4 = 16
    """``ReservedBit4`` in MLIR text."""

    VIRTUAL = 32
    """``Virtual`` in MLIR text."""

    ARTIFICIAL = 64
    """``Artificial`` in MLIR text."""

    EXPLICIT = 128
    """``Explicit`` in MLIR text."""

    PROTOTYPED = 256
    """``Prototyped`` in MLIR text."""

    OBJC_CLASS_COMPLETE = 512
    """``ObjcClassComplete`` in MLIR text."""

    OBJECT_POINTER = 1024
    """``ObjectPointer`` in MLIR text."""

    VECTOR = 2048
    """``Vector`` in MLIR text."""

    STATIC_MEMBER = 4096
    """``StaticMember`` in MLIR text."""

    LVALUE_REFERENCE = 8192
    """``LValueReference`` in MLIR text."""

    RVALUE_REFERENCE = 16384
    """``RValueReference`` in MLIR text."""

    EXPORT_SYMBOLS = 32768
    """``ExportSymbols`` in MLIR text."""

    SINGLE_INHERITANCE = 65536
    """``SingleInheritance`` in MLIR text."""

    MULTIPLE_INHERITANCE = 65536
    """``MultipleInheritance`` in MLIR text."""

    VIRTUAL_INHERITANCE = 65536
    """``VirtualInheritance`` in MLIR text."""

    INTRODUCED_VIRTUAL = 262144
    """``IntroducedVirtual`` in MLIR text."""

    BIT_FIELD = 524288
    """``BitField`` in MLIR text."""

    NO_RETURN = 1048576
    """``NoReturn`` in MLIR text."""

    TYPE_PASS_BY_VALUE = 4194304
    """``TypePassByValue`` in MLIR text."""

    TYPE_PASS_BY_REFERENCE = 8388608
    """``TypePassByReference`` in MLIR text."""

    ENUM_CLASS = 16777216
    """``EnumClass`` in MLIR text."""

    THUNK = 33554432
    """``Thunk`` in MLIR text."""

    NON_TRIVIAL = 67108864
    """``NonTrivial`` in MLIR text."""

    BIG_ENDIAN = 134217728
    """``BigEndian`` in MLIR text."""

    LITTLE_ENDIAN = 268435456
    """``LittleEndian`` in MLIR text."""

    ALL_CALLS_DESCRIBED = 536870912
    """``AllCallsDescribed`` in MLIR text."""

class DISubprogramFlags(enum.Flag):
    """LLVM DISubprogram flags"""

    VIRTUAL = 1
    """``Virtual`` in MLIR text."""

    PURE_VIRTUAL = 2
    """``PureVirtual`` in MLIR text."""

    LOCAL_TO_UNIT = 4
    """``LocalToUnit`` in MLIR text."""

    DEFINITION = 8
    """``Definition`` in MLIR text."""

    OPTIMIZED = 16
    """``Optimized`` in MLIR text."""

    PURE = 32
    """``Pure`` in MLIR text."""

    ELEMENTAL = 64
    """``Elemental`` in MLIR text."""

    RECURSIVE = 128
    """``Recursive`` in MLIR text."""

    MAIN_SUBPROGRAM = 256
    """``MainSubprogram`` in MLIR text."""

    DELETED = 512
    """``Deleted`` in MLIR text."""

    OBJ_CDIRECT = 2048
    """``ObjCDirect`` in MLIR text."""

class FPExceptionBehavior(enum.Enum):
    """LLVM Exception Behavior"""

    IGNORE = 0
    """``ignore`` in MLIR text."""

    MAY_TRAP = 1
    """``maytrap`` in MLIR text."""

    STRICT = 2
    """``strict`` in MLIR text."""

class DIEmissionKind(enum.Enum):
    """LLVM debug emission kind"""

    NONE = 0
    """``None`` in MLIR text."""

    FULL = 1
    """``Full`` in MLIR text."""

    LINE_TABLES_ONLY = 2
    """``LineTablesOnly`` in MLIR text."""

    DEBUG_DIRECTIVES_ONLY = 3
    """``DebugDirectivesOnly`` in MLIR text."""

class DINameTableKind(enum.Enum):
    """LLVM debug name table kind"""

    DEFAULT = 0
    """``Default`` in MLIR text."""

    GNU = 1
    """``GNU`` in MLIR text."""

    NONE = 2
    """``None`` in MLIR text."""

    APPLE = 3
    """``Apple`` in MLIR text."""

class ProfileSummaryFormatKind(enum.Enum):
    """LLVM ProfileSummary format kinds"""

    SAMPLE_PROFILE = 0
    """``SampleProfile`` in MLIR text."""

    INSTR_PROF = 1
    """``InstrProf`` in MLIR text."""

    CSINSTR_PROF = 2
    """``CSInstrProf`` in MLIR text."""

class ModFlagBehavior(enum.Enum):
    """LLVM Module Flag Behavior"""

    ERROR = 1
    """``error`` in MLIR text."""

    WARNING = 2
    """``warning`` in MLIR text."""

    REQUIRE = 3
    """``require`` in MLIR text."""

    OVERRIDE = 4
    """``override`` in MLIR text."""

    APPEND = 5
    """``append`` in MLIR text."""

    APPEND_UNIQUE = 6
    """``append_unique`` in MLIR text."""

    MAX = 7
    """``max`` in MLIR text."""

    MIN = 8
    """``min`` in MLIR text."""

class ModRefInfo(enum.Enum):
    """LLVM ModRefInfo"""

    NO_MOD_REF = 0
    """``none`` in MLIR text."""

    REF = 1
    """``read`` in MLIR text."""

    MOD = 2
    """``write`` in MLIR text."""

    MOD_REF = 3
    """``readwrite`` in MLIR text."""

class RoundingMode(enum.Enum):
    """LLVM Rounding Mode"""

    TOWARD_ZERO = 0
    """``towardzero`` in MLIR text."""

    NEAREST_TIES_TO_EVEN = 1
    """``tonearest`` in MLIR text."""

    TOWARD_POSITIVE = 2
    """``upward`` in MLIR text."""

    TOWARD_NEGATIVE = 3
    """``downward`` in MLIR text."""

    NEAREST_TIES_TO_AWAY = 4
    """``tonearestaway`` in MLIR text."""

    DYNAMIC = 7
    """``dynamic`` in MLIR text."""

    INVALID = 8
    """``invalid`` in MLIR text."""

class AShrOp(mlir_python._mlir_python.Operation):
    """``llvm.ashr``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.ashr``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.ashr"

class AddOp(mlir_python._mlir_python.Operation):
    """``llvm.add``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.add``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            overflow_flags: Property ``overflowFlags`` (flags of ``IntegerOverflowFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Property ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.add"

class AddrSpaceCastOp(mlir_python._mlir_python.Operation):
    """``llvm.addrspacecast``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.addrspacecast``.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            arg: Operand ``arg`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    OPERATION_NAME: str = "llvm.addrspacecast"

class AddressOfOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.addressof``: Creates a pointer pointing to a global, alias or a function.

    Creates an SSA value containing a pointer to a global value (function,
    variable or alias). The global value can be defined after its first
    referenced. If the global value is a constant, storing into it is not
    allowed.

    Examples:

    ```mlir
    func @foo() {
      // Get the address of a global variable.
      %0 = llvm.mlir.addressof @const : !llvm.ptr

      // Use it as a regular pointer.
      %1 = llvm.load %0 : !llvm.ptr -> i32

      // Get the address of a function.
      %2 = llvm.mlir.addressof @foo : !llvm.ptr

      // The function address can be used for indirect calls.
      llvm.call %2() : !llvm.ptr, () -> ()

      // Get the address of an aliased global.
      %3 = llvm.mlir.addressof @const_alias : !llvm.ptr
    }

    // Define the global.
    llvm.mlir.global @const(42 : i32) : i32

    // Define an alias.
    llvm.mlir.alias @const_alias : i32 {
      %0 = llvm.mlir.addressof @const : !llvm.ptr
      llvm.return %0 : !llvm.ptr
    }
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        global_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.addressof``: Creates a pointer pointing to a global, alias or a function.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type).
            global_name: Attribute ``global_name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM pointer type."""

    @property
    def global_name(self) -> str:
        """Attribute ``global_name``: flat symbol reference attribute."""

    @global_name.setter
    def global_name(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "llvm.mlir.addressof"

class AliasOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.alias``: LLVM dialect alias..

    `llvm.mlir.alias` is a top level operation that defines a global alias for
    global variables and functions. The operation is always initialized by
    using a initializer region which could be a direct map to another global
    value or contain some address computation on top of it.

    It uses a symbol for its value, which will be uniqued by the module
    with respect to other symbols in it.

    Similarly to functions and globals, they can also have a linkage attribute.
    This attribute is placed between `llvm.mlir.alias` and the symbol name. If
    the attribute is omitted, `external` linkage is assumed by default.

    Examples:

    ```mlir
    // Global alias use @-identifiers.
    llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
      %0 = llvm.mlir.addressof @some_function : !llvm.ptr
      llvm.return %0 : !llvm.ptr
    }

    // More complex initialization.
    llvm.mlir.alias linkonce_odr hidden @glob
    {addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
      %0 = llvm.mlir.constant(1234 : i64) : i64
      %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
      %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
      %3 = llvm.add %2, %0 : i64
      %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
      llvm.return %4 : !llvm.ptr
    }
    ```
    """

    def __init__(
        self,
        alias_type: mlir_python._mlir_python.Type,
        sym_name: str,
        linkage: Linkage,
        *,
        dso_local: bool = False,
        thread_local_: bool = False,
        unnamed_addr: UnnamedAddr | None = None,
        visibility_: Visibility = Visibility.DEFAULT,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.alias``: LLVM dialect alias..

        Args:
            alias_type: Attribute ``alias_type`` (any type attribute).
            sym_name: Attribute ``sym_name`` (string attribute).
            linkage: Attribute ``linkage`` (a member of ``Linkage``).
            dso_local: Attribute ``dso_local`` (unit attribute). Omit for the default.
            thread_local_: Attribute ``thread_local_`` (unit attribute). Omit for the default.
            unnamed_addr: Attribute ``unnamed_addr`` (a member of ``UnnamedAddr``). Optional.
            visibility_: Attribute ``visibility_`` (a member of ``Visibility``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def alias_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``alias_type``: any type attribute."""

    @alias_type.setter
    def alias_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def linkage(self) -> Linkage:
        """Attribute ``linkage``: a member of ``Linkage``."""

    @linkage.setter
    def linkage(self, arg: Linkage, /) -> None: ...
    @property
    def dso_local(self) -> bool:
        """Attribute ``dso_local``: unit attribute."""

    @dso_local.setter
    def dso_local(self, arg: bool, /) -> None: ...
    @property
    def unnamed_addr(self) -> UnnamedAddr | None:
        """Attribute ``unnamed_addr``: a member of ``UnnamedAddr``."""

    @unnamed_addr.setter
    def unnamed_addr(self, arg: UnnamedAddr | None) -> None: ...
    @property
    def initializer(self) -> mlir_python._mlir_python.Region:
        """Region ``initializer``: region with 1 blocks."""

    OPERATION_NAME: str = "llvm.mlir.alias"

class AllocaOp(mlir_python._mlir_python.Operation):
    """``llvm.alloca``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        array_size: mlir_python._mlir_python.Value,
        elem_type: mlir_python._mlir_python.Type,
        *,
        alignment: int | None = None,
        inalloca: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.alloca``.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type).
            array_size: Operand ``arraySize`` (signless integer).
            elem_type: Attribute ``elem_type`` (any type attribute).
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            inalloca: Attribute ``inalloca`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def array_size(self) -> mlir_python._mlir_python.Value:
        """Operand ``arraySize``: signless integer."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM pointer type."""

    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def elem_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``elem_type``: any type attribute."""

    @elem_type.setter
    def elem_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def inalloca(self) -> bool:
        """Attribute ``inalloca``: unit attribute."""

    @inalloca.setter
    def inalloca(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.alloca"

class AndOp(mlir_python._mlir_python.Operation):
    """``llvm.and``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.and``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.and"

class AtomicCmpXchgOp(mlir_python._mlir_python.Operation):
    """``llvm.cmpxchg``."""

    def __init__(
        self,
        ptr: mlir_python._mlir_python.Value,
        cmp: mlir_python._mlir_python.Value,
        val: mlir_python._mlir_python.Value,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        *,
        syncscope: str | None = None,
        alignment: int | None = None,
        weak: bool = False,
        volatile_: bool = False,
        access_groups: mlir_python._mlir_python.ArrayAttr | None = None,
        alias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        noalias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        tbaa: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.cmpxchg``.

        Result types are inferred.

        Args:
            ptr: Operand ``ptr`` (LLVM pointer type).
            cmp: Operand ``cmp`` (signless integer or LLVM pointer type).
            val: Operand ``val`` (signless integer or LLVM pointer type).
            success_ordering: Attribute ``success_ordering`` (a member of ``AtomicOrdering``).
            failure_ordering: Attribute ``failure_ordering`` (a member of ``AtomicOrdering``).
            syncscope: Attribute ``syncscope`` (string attribute). Optional.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            weak: Attribute ``weak`` (unit attribute). Omit for the default.
            volatile_: Attribute ``volatile_`` (unit attribute). Omit for the default.
            access_groups: Attribute ``access_groups`` (LLVM dialect access group metadata array). Optional.
            alias_scopes: Attribute ``alias_scopes`` (LLVM dialect alias scope array). Optional.
            noalias_scopes: Attribute ``noalias_scopes`` (LLVM dialect alias scope array). Optional.
            tbaa: Attribute ``tbaa`` (LLVM dialect TBAA tag metadata array). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def ptr(self) -> mlir_python._mlir_python.Value:
        """Operand ``ptr``: LLVM pointer type."""

    @property
    def cmp(self) -> mlir_python._mlir_python.Value:
        """Operand ``cmp``: signless integer or LLVM pointer type."""

    @property
    def val(self) -> mlir_python._mlir_python.Value:
        """Operand ``val``: signless integer or LLVM pointer type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM structure type."""

    @property
    def success_ordering(self) -> AtomicOrdering:
        """Attribute ``success_ordering``: a member of ``AtomicOrdering``."""

    @success_ordering.setter
    def success_ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def failure_ordering(self) -> AtomicOrdering:
        """Attribute ``failure_ordering``: a member of ``AtomicOrdering``."""

    @failure_ordering.setter
    def failure_ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def syncscope(self) -> str | None:
        """Attribute ``syncscope``: string attribute."""

    @syncscope.setter
    def syncscope(self, arg: str | None) -> None: ...
    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def weak(self) -> bool:
        """Attribute ``weak``: unit attribute."""

    @weak.setter
    def weak(self, arg: bool, /) -> None: ...
    @property
    def access_groups(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``access_groups``: LLVM dialect access group metadata array."""

    @access_groups.setter
    def access_groups(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def alias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``alias_scopes``: LLVM dialect alias scope array."""

    @alias_scopes.setter
    def alias_scopes(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def noalias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``noalias_scopes``: LLVM dialect alias scope array."""

    @noalias_scopes.setter
    def noalias_scopes(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def tbaa(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``tbaa``: LLVM dialect TBAA tag metadata array."""

    @tbaa.setter
    def tbaa(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.cmpxchg"

class AtomicRMWOp(mlir_python._mlir_python.Operation):
    """``llvm.atomicrmw``."""

    def __init__(
        self,
        bin_op: AtomicBinOp,
        ptr: mlir_python._mlir_python.Value,
        val: mlir_python._mlir_python.Value,
        ordering: AtomicOrdering,
        *,
        syncscope: str | None = None,
        alignment: int | None = None,
        volatile_: bool = False,
        access_groups: mlir_python._mlir_python.ArrayAttr | None = None,
        alias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        noalias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        tbaa: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.atomicrmw``.

        Result types are inferred.

        Args:
            bin_op: Attribute ``bin_op`` (a member of ``AtomicBinOp``).
            ptr: Operand ``ptr`` (LLVM pointer type).
            val: Operand ``val`` (floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type).
            ordering: Attribute ``ordering`` (a member of ``AtomicOrdering``).
            syncscope: Attribute ``syncscope`` (string attribute). Optional.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            volatile_: Attribute ``volatile_`` (unit attribute). Omit for the default.
            access_groups: Attribute ``access_groups`` (LLVM dialect access group metadata array). Optional.
            alias_scopes: Attribute ``alias_scopes`` (LLVM dialect alias scope array). Optional.
            noalias_scopes: Attribute ``noalias_scopes`` (LLVM dialect alias scope array). Optional.
            tbaa: Attribute ``tbaa`` (LLVM dialect TBAA tag metadata array). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def ptr(self) -> mlir_python._mlir_python.Value:
        """Operand ``ptr``: LLVM pointer type."""

    @property
    def val(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``val``: floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type.
        """

    @property
    def bin_op(self) -> AtomicBinOp:
        """Attribute ``bin_op``: a member of ``AtomicBinOp``."""

    @bin_op.setter
    def bin_op(self, arg: AtomicBinOp, /) -> None: ...
    @property
    def ordering(self) -> AtomicOrdering:
        """Attribute ``ordering``: a member of ``AtomicOrdering``."""

    @ordering.setter
    def ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def syncscope(self) -> str | None:
        """Attribute ``syncscope``: string attribute."""

    @syncscope.setter
    def syncscope(self, arg: str | None) -> None: ...
    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def access_groups(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``access_groups``: LLVM dialect access group metadata array."""

    @access_groups.setter
    def access_groups(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def alias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``alias_scopes``: LLVM dialect alias scope array."""

    @alias_scopes.setter
    def alias_scopes(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def noalias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``noalias_scopes``: LLVM dialect alias scope array."""

    @noalias_scopes.setter
    def noalias_scopes(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def tbaa(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``tbaa``: LLVM dialect TBAA tag metadata array."""

    @tbaa.setter
    def tbaa(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.atomicrmw"

class BitcastOp(mlir_python._mlir_python.Operation):
    """``llvm.bitcast``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.bitcast``.

        Args:
            res_type: Type of result ``res`` (LLVM-compatible non-aggregate type).
            arg: Operand ``arg`` (LLVM-compatible non-aggregate type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """Operand ``arg``: LLVM-compatible non-aggregate type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM-compatible non-aggregate type."""

    OPERATION_NAME: str = "llvm.bitcast"

class BlockAddressOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.blockaddress``: Creates a LLVM blockaddress ptr.

    Creates an SSA value containing a pointer to a basic block. The block
    address information (function and block) is given by the `BlockAddressAttr`
    attribute. This operation assumes an existing `llvm.blocktag` operation
    identifying an existing MLIR block within a function. Example:

    ```mlir
    llvm.mlir.global private @g() : !llvm.ptr {
      %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
      llvm.return %0 : !llvm.ptr
    }

    llvm.func @fn() {
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      llvm.blocktag <id = 0>
      llvm.return
    }
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        block_addr: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.blockaddress``: Creates a LLVM blockaddress ptr.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type).
            block_addr: Attribute ``block_addr``.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM pointer type."""

    @property
    def block_addr(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``block_addr``."""

    @block_addr.setter
    def block_addr(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.blockaddress"

class BlockTagOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.blocktag``.

    This operation uses a `tag` to uniquely identify an MLIR block in a
    function. The same tag is used by `llvm.blockaddress` in order to compute
    the target address.

    A given function should have at most one `llvm.blocktag` operation with a
    given `tag`. This operation cannot be used as a terminator.

    Example:

    ```mlir
    llvm.func @f() -> !llvm.ptr {
      %addr = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
      llvm.br ^bb1
    ^bb1:
      llvm.blocktag <id = 1>
      llvm.return %addr : !llvm.ptr
    }
    ```
    """

    def __init__(
        self,
        tag: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.blocktag``.

        Args:
            tag: Attribute ``tag``.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def tag(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``tag``."""

    @tag.setter
    def tag(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.blocktag"

class BrOp(mlir_python._mlir_python.Operation):
    """``llvm.br``."""

    def __init__(
        self,
        dest: mlir_python._mlir_python.Block,
        dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        loop_annotation: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.br``.

        Args:
            dest: Successor ``dest`` (any successor).
            dest_operands: Operand ``destOperands`` (LLVM dialect-compatible type). Empty by default.
            loop_annotation: Attribute ``loop_annotation``. Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``destOperands``: LLVM dialect-compatible type."""

    @property
    def loop_annotation(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``loop_annotation``."""

    @loop_annotation.setter
    def loop_annotation(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``dest``."""

    OPERATION_NAME: str = "llvm.br"

class CallIntrinsicOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.call_intrinsic``: Call to an LLVM intrinsic function..

    Call the specified llvm intrinsic. If the intrinsic is overloaded, use
    the MLIR function type of this op to determine which intrinsic to call.
    """

    def __init__(
        self,
        intrin: str,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        op_bundle_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        op_bundle_tags: mlir_python._mlir_python.ArrayAttr | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        result_type: mlir_python._mlir_python.Type | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.call_intrinsic``: Call to an LLVM intrinsic function..

        Args:
            intrin: Attribute ``intrin`` (string attribute).
            args: Operand ``args`` (LLVM dialect-compatible type). Empty by default.
            op_bundle_operands: Operand ``op_bundle_operands`` (LLVM dialect-compatible type). Empty by default.
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            op_bundle_tags: Attribute ``op_bundle_tags`` (array attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            result_type: Type of result ``results`` (LLVM dialect-compatible type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: LLVM dialect-compatible type."""

    @property
    def op_bundle_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``op_bundle_operands``: LLVM dialect-compatible type."""

    @property
    def intrin(self) -> str:
        """Attribute ``intrin``: string attribute."""

    @intrin.setter
    def intrin(self, arg: str, /) -> None: ...
    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...
    @property
    def op_bundle_tags(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``op_bundle_tags``: array attribute."""

    @op_bundle_tags.setter
    def op_bundle_tags(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.call_intrinsic"

class CallOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.call``: Call to an LLVM function..

    In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
    implements this behavior by providing a variadic `call` operation for 0- and
    1-result functions. Even though MLIR supports multi-result functions, LLVM
    IR dialect disallows them.

    The `call` instruction supports both direct and indirect calls. Direct calls
    start with a function name (`@`-prefixed) and indirect calls start with an
    SSA value (`%`-prefixed). The direct callee, if present, is stored as a
    function attribute `callee`. For indirect calls, the callee is of `!llvm.ptr` type
    and is stored as the first value in `callee_operands`. If and only if the
    callee is a variadic function, the `var_callee_type` attribute must carry
    the variadic LLVM function type. The trailing type list contains the
    optional indirect callee type and the MLIR function type, which differs from
    the LLVM function type that uses an explicit void type to model functions
    that do not return a value.

    If this operatin has the `no_inline` attribute, then this specific function call
    will never be inlined. The opposite behavior will occur if the call has `always_inline`
    attribute. The `inline_hint` attribute indicates that it is desirable to inline
    this function call.

    Examples:

    ```mlir
    // Direct call without arguments and with one result.
    %0 = llvm.call @foo() : () -> (f32)

    // Direct call with arguments and without a result.
    llvm.call @bar(%0) : (f32) -> ()

    // Indirect call with an argument and without a result.
    %1 = llvm.mlir.addressof @foo : !llvm.ptr
    llvm.call %1(%0) : !llvm.ptr, (f32) -> ()

    // Direct variadic call.
    llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    // Indirect variadic call
    llvm.call %1(%0) vararg(!llvm.func<void (...)>) : !llvm.ptr, (i32) -> ()
    ```
    """

    def __init__(
        self,
        callee_operands: Sequence[mlir_python._mlir_python.Value] = [],
        op_bundle_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        var_callee_type: mlir_python._mlir_python.Type | None = None,
        callee: str | None = None,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        cconv: mlir_python._mlir_python.Attribute | None = None,
        tail_call_kind: mlir_python._mlir_python.Attribute | None = None,
        memory_effects: mlir_python._mlir_python.Attribute | None = None,
        convergent: bool = False,
        no_unwind: bool = False,
        will_return: bool = False,
        op_bundle_tags: mlir_python._mlir_python.ArrayAttr | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        no_inline: bool = False,
        always_inline: bool = False,
        inline_hint: bool = False,
        access_groups: mlir_python._mlir_python.ArrayAttr | None = None,
        alias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        noalias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        tbaa: mlir_python._mlir_python.ArrayAttr | None = None,
        result_type: mlir_python._mlir_python.Type | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.call``: Call to an LLVM function..

        Args:
            callee_operands: Operand ``callee_operands`` (LLVM dialect-compatible type). Empty by default.
            op_bundle_operands: Operand ``op_bundle_operands`` (LLVM dialect-compatible type). Empty by default.
            var_callee_type: Attribute ``var_callee_type`` (type attribute of LLVM function type). Optional.
            callee: Attribute ``callee`` (flat symbol reference attribute). Optional.
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            cconv: Attribute ``CConv`` (LLVM Calling Convention specification). Omit for the default.
            tail_call_kind: Attribute ``TailCallKind`` (LLVM Calling Convention specification). Omit for the default.
            memory_effects: Attribute ``memory_effects``. Optional.
            convergent: Attribute ``convergent`` (unit attribute). Omit for the default.
            no_unwind: Attribute ``no_unwind`` (unit attribute). Omit for the default.
            will_return: Attribute ``will_return`` (unit attribute). Omit for the default.
            op_bundle_tags: Attribute ``op_bundle_tags`` (array attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            no_inline: Attribute ``no_inline`` (unit attribute). Omit for the default.
            always_inline: Attribute ``always_inline`` (unit attribute). Omit for the default.
            inline_hint: Attribute ``inline_hint`` (unit attribute). Omit for the default.
            access_groups: Attribute ``access_groups`` (LLVM dialect access group metadata array). Optional.
            alias_scopes: Attribute ``alias_scopes`` (LLVM dialect alias scope array). Optional.
            noalias_scopes: Attribute ``noalias_scopes`` (LLVM dialect alias scope array). Optional.
            tbaa: Attribute ``tbaa`` (LLVM dialect TBAA tag metadata array). Optional.
            result_type: Type of result ``result`` (LLVM dialect-compatible type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``callee_operands``: LLVM dialect-compatible type."""

    @property
    def op_bundle_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``op_bundle_operands``: LLVM dialect-compatible type."""

    @property
    def var_callee_type(self) -> mlir_python._mlir_python.Type | None:
        """Attribute ``var_callee_type``: type attribute of LLVM function type."""

    @var_callee_type.setter
    def var_callee_type(self, arg: mlir_python._mlir_python.Type | None) -> None: ...
    @property
    def callee(self) -> str | None:
        """Attribute ``callee``: flat symbol reference attribute."""

    @callee.setter
    def callee(self, arg: str | None) -> None: ...
    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...
    @property
    def cconv(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``CConv``: LLVM Calling Convention specification."""

    @cconv.setter
    def cconv(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def tail_call_kind(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``TailCallKind``: LLVM Calling Convention specification."""

    @tail_call_kind.setter
    def tail_call_kind(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def memory_effects(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``memory_effects``."""

    @memory_effects.setter
    def memory_effects(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def convergent(self) -> bool:
        """Attribute ``convergent``: unit attribute."""

    @convergent.setter
    def convergent(self, arg: bool, /) -> None: ...
    @property
    def no_unwind(self) -> bool:
        """Attribute ``no_unwind``: unit attribute."""

    @no_unwind.setter
    def no_unwind(self, arg: bool, /) -> None: ...
    @property
    def will_return(self) -> bool:
        """Attribute ``will_return``: unit attribute."""

    @will_return.setter
    def will_return(self, arg: bool, /) -> None: ...
    @property
    def op_bundle_tags(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``op_bundle_tags``: array attribute."""

    @op_bundle_tags.setter
    def op_bundle_tags(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def no_inline(self) -> bool:
        """Attribute ``no_inline``: unit attribute."""

    @no_inline.setter
    def no_inline(self, arg: bool, /) -> None: ...
    @property
    def always_inline(self) -> bool:
        """Attribute ``always_inline``: unit attribute."""

    @always_inline.setter
    def always_inline(self, arg: bool, /) -> None: ...
    @property
    def inline_hint(self) -> bool:
        """Attribute ``inline_hint``: unit attribute."""

    @inline_hint.setter
    def inline_hint(self, arg: bool, /) -> None: ...
    @property
    def access_groups(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``access_groups``: LLVM dialect access group metadata array."""

    @access_groups.setter
    def access_groups(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def alias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``alias_scopes``: LLVM dialect alias scope array."""

    @alias_scopes.setter
    def alias_scopes(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def noalias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``noalias_scopes``: LLVM dialect alias scope array."""

    @noalias_scopes.setter
    def noalias_scopes(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def tbaa(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``tbaa``: LLVM dialect TBAA tag metadata array."""

    @tbaa.setter
    def tbaa(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.call"

class ComdatOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.comdat``: LLVM dialect comdat region.

    Provides access to object file COMDAT section/group functionality.

    Examples:
    ```mlir
    llvm.comdat @__llvm_comdat {
      llvm.comdat_selector @any any
    }
    llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
    ```
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.comdat``: LLVM dialect comdat region.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "llvm.comdat"

class ComdatSelectorOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.comdat_selector``: LLVM dialect comdat selector declaration.

    Provides access to object file COMDAT section/group functionality.

    Examples:
    ```mlir
    llvm.comdat @__llvm_comdat {
      llvm.comdat_selector @any any
    }
    llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
    ```
    """

    def __init__(
        self,
        sym_name: str,
        comdat: Comdat,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.comdat_selector``: LLVM dialect comdat selector declaration.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            comdat: Attribute ``comdat`` (a member of ``Comdat``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def comdat(self) -> Comdat:
        """Attribute ``comdat``: a member of ``Comdat``."""

    @comdat.setter
    def comdat(self, arg: Comdat, /) -> None: ...

    OPERATION_NAME: str = "llvm.comdat_selector"

class CondBrOp(mlir_python._mlir_python.Operation):
    """``llvm.cond_br``."""

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        true_dest: mlir_python._mlir_python.Block,
        false_dest: mlir_python._mlir_python.Block,
        true_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        false_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        branch_weights: mlir_python._mlir_python.Attribute | None = None,
        loop_annotation: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.cond_br``.

        Args:
            condition: Operand ``condition`` (1-bit signless integer).
            true_dest: Successor ``trueDest`` (any successor).
            false_dest: Successor ``falseDest`` (any successor).
            true_dest_operands: Operand ``trueDestOperands`` (LLVM dialect-compatible type). Empty by default.
            false_dest_operands: Operand ``falseDestOperands`` (LLVM dialect-compatible type). Empty by default.
            branch_weights: Attribute ``branch_weights`` (i32 dense array attribute). Optional.
            loop_annotation: Attribute ``loop_annotation``. Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def true_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``trueDestOperands``: LLVM dialect-compatible type."""

    @property
    def false_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``falseDestOperands``: LLVM dialect-compatible type."""

    @property
    def branch_weights(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``branch_weights``: i32 dense array attribute."""

    @branch_weights.setter
    def branch_weights(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def loop_annotation(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``loop_annotation``."""

    @loop_annotation.setter
    def loop_annotation(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def true_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``trueDest``."""

    @property
    def false_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``falseDest``."""

    OPERATION_NAME: str = "llvm.cond_br"

class ConstantOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.constant``: Defines a constant of LLVM type..

    Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
    all constants must be created as SSA values before being used in other
    operations. `llvm.mlir.constant` creates such values for scalars, vectors,
    strings, structs, and array of structs. It has a mandatory `value` attribute
    whose type depends on the type of the constant value. The type of the constant
    value must correspond to the attribute type converted to LLVM IR type.

    When creating constant scalars, the `value` attribute must be either an
    integer attribute or a floating point attribute. The type of the attribute
    may be omitted for `i64` and `f64` types that are implied.

    When creating constant vectors, the `value` attribute must be either an
    array attribute, a dense attribute, or a sparse attribute that contains
    integers or floats. The number of elements in the result vector must match
    the number of elements in the attribute.

    When creating constant strings, the `value` attribute must be a string
    attribute. The type of the constant must be an LLVM array of `i8`s, and the
    length of the array must match the length of the attribute.

    When creating constant structs, the `value` attribute must be an array
    attribute that contains integers or floats. The type of the constant must be
    an LLVM struct type. The number of fields in the struct must match the
    number of elements in the attribute, and the type of each LLVM struct field
    must correspond to the type of the corresponding attribute element converted
    to LLVM IR.

    When creating an array of structs, the `value` attribute must be an array
    attribute, itself containing zero, or undef, or array attributes for each
    potential nested array type, and the elements of the leaf array attributes
    for must match the struct element types or be zero or undef attributes.

    Examples:

    ```mlir
    // Integer constant, internal i32 is mandatory
    %0 = llvm.mlir.constant(42 : i32) : i32

    // It's okay to omit i64.
    %1 = llvm.mlir.constant(42) : i64

    // Floating point constant.
    %2 = llvm.mlir.constant(42.0 : f32) : f32

    // Splat dense vector constant.
    %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        value: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.constant``: Defines a constant of LLVM type..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            value: Attribute ``value`` (any attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    @property
    def value(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``value``: any attribute."""

    @value.setter
    def value(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.mlir.constant"

class DSOLocalEquivalentOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.dso_local_equivalent``: Creates a LLVM dso_local_equivalent ptr.

    Creates an SSA value containing a pointer to a global value (function or
    alias to function). It represents a function which is functionally
    equivalent to a given function, but is always defined in the current
    linkage unit. The target function may not have `extern_weak` linkage.

    Examples:

    ```mlir
    llvm.mlir.global external constant @const() : i64 {
      %0 = llvm.mlir.addressof @const : !llvm.ptr
      %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
      %2 = llvm.dso_local_equivalent @func : !llvm.ptr
      %4 = llvm.ptrtoint %2 : !llvm.ptr to i64
      llvm.return %4 : i64
    }
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        function_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.dso_local_equivalent``: Creates a LLVM dso_local_equivalent ptr.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type).
            function_name: Attribute ``function_name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM pointer type."""

    @property
    def function_name(self) -> str:
        """Attribute ``function_name``: flat symbol reference attribute."""

    @function_name.setter
    def function_name(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "llvm.dso_local_equivalent"

class ExtractElementOp(mlir_python._mlir_python.Operation):
    """``llvm.extractelement``: Extract an element from an LLVM vector.."""

    def __init__(
        self,
        vector: mlir_python._mlir_python.Value,
        position: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.extractelement``: Extract an element from an LLVM vector..

        Result types are inferred.

        Args:
            vector: Operand ``vector`` (LLVM dialect-compatible vector type).
            position: Operand ``position`` (signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def vector(self) -> mlir_python._mlir_python.Value:
        """Operand ``vector``: LLVM dialect-compatible vector type."""

    @property
    def position(self) -> mlir_python._mlir_python.Value:
        """Operand ``position``: signless integer."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.extractelement"

class ExtractValueOp(mlir_python._mlir_python.Operation):
    """``llvm.extractvalue``: Extract a value from an LLVM struct.."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        container: mlir_python._mlir_python.Value,
        position: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.extractvalue``: Extract a value from an LLVM struct..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            container: Operand ``container`` (LLVM aggregate type).
            position: Attribute ``position`` (i64 dense array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def container(self) -> mlir_python._mlir_python.Value:
        """Operand ``container``: LLVM aggregate type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    @property
    def position(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``position``: i64 dense array attribute."""

    @position.setter
    def position(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.extractvalue"

class FAddOp(mlir_python._mlir_python.Operation):
    """``llvm.fadd``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fadd``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fadd"

class FCmpOp(mlir_python._mlir_python.Operation):
    """``llvm.fcmp``."""

    def __init__(
        self,
        predicate: FCmpPredicate,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fcmp``.

        Result types are inferred.

        Args:
            predicate: Attribute ``predicate`` (a member of ``FCmpPredicate``).
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer.
        """

    @property
    def predicate(self) -> FCmpPredicate:
        """Attribute ``predicate``: a member of ``FCmpPredicate``."""

    @predicate.setter
    def predicate(self, arg: FCmpPredicate, /) -> None: ...
    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fcmp"

class FDivOp(mlir_python._mlir_python.Operation):
    """``llvm.fdiv``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fdiv``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fdiv"

class FMulOp(mlir_python._mlir_python.Operation):
    """``llvm.fmul``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fmul``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fmul"

class FNegOp(mlir_python._mlir_python.Operation):
    """``llvm.fneg``."""

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fneg``.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fneg"

class FPExtOp(mlir_python._mlir_python.Operation):
    """``llvm.fpext``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fpext``.

        Args:
            res_type: Type of result ``res`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            arg: Operand ``arg`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    OPERATION_NAME: str = "llvm.fpext"

class FPToSIOp(mlir_python._mlir_python.Operation):
    """``llvm.fptosi``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fptosi``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.fptosi"

class FPToUIOp(mlir_python._mlir_python.Operation):
    """``llvm.fptoui``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fptoui``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.fptoui"

class FPTruncOp(mlir_python._mlir_python.Operation):
    """``llvm.fptrunc``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fptrunc``.

        Args:
            res_type: Type of result ``res`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            arg: Operand ``arg`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    OPERATION_NAME: str = "llvm.fptrunc"

class FRemOp(mlir_python._mlir_python.Operation):
    """``llvm.frem``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.frem``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.frem"

class FSubOp(mlir_python._mlir_python.Operation):
    """``llvm.fsub``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fsub``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            rhs: Operand ``rhs`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.fsub"

class FenceOp(mlir_python._mlir_python.Operation):
    """``llvm.fence``."""

    def __init__(
        self,
        ordering: AtomicOrdering,
        *,
        syncscope: str | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.fence``.

        Args:
            ordering: Attribute ``ordering`` (a member of ``AtomicOrdering``).
            syncscope: Attribute ``syncscope`` (string attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def ordering(self) -> AtomicOrdering:
        """Attribute ``ordering``: a member of ``AtomicOrdering``."""

    @ordering.setter
    def ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def syncscope(self) -> str | None:
        """Attribute ``syncscope``: string attribute."""

    @syncscope.setter
    def syncscope(self, arg: str | None) -> None: ...

    OPERATION_NAME: str = "llvm.fence"

class FreezeOp(mlir_python._mlir_python.Operation):
    """``llvm.freeze``."""

    def __init__(
        self,
        val: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.freeze``.

        Result types are inferred.

        Args:
            val: Operand ``val`` (LLVM dialect-compatible type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def val(self) -> mlir_python._mlir_python.Value:
        """Operand ``val``: LLVM dialect-compatible type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.freeze"

class GEPOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.getelementptr``.

    This operation mirrors LLVM IRs 'getelementptr' operation that is used to
    perform pointer arithmetic.

    Like in LLVM IR, it is possible to use both constants as well as SSA values
    as indices. In the case of indexing within a structure, it is required to
    either use constant indices directly, or supply a constant SSA value.

    The no-wrap flags can be used to specify the low-level pointer arithmetic
    overflow behavior that LLVM uses after lowering the operation to LLVM IR.
    Valid options include 'inbounds' (pointer arithmetic must be within object
    bounds), 'nusw' (no unsigned signed wrap), and 'nuw' (no unsigned wrap).
    Note that 'inbounds' implies 'nusw' which is ensured by the enum
    definition. The flags can be set individually or in combination.

    Examples:

    ```mlir
    // GEP with an SSA value offset
    %0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

    // GEP with a constant offset and the inbounds attribute set
    %0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -> !llvm.ptr, f32

    // GEP with constant offsets into a structure
    %0 = llvm.getelementptr %1[0, 1]
       : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f32)>
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        base: mlir_python._mlir_python.Value,
        raw_constant_indices: mlir_python._mlir_python.Attribute,
        elem_type: mlir_python._mlir_python.Type,
        dynamic_indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        no_wrap_flags: GEPNoWrapFlags = GEPNoWrapFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.getelementptr``.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            base: Operand ``base`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            raw_constant_indices: Attribute ``rawConstantIndices`` (i32 dense array attribute).
            elem_type: Attribute ``elem_type`` (any type attribute).
            dynamic_indices: Operand ``dynamicIndices`` (signless integer or LLVM dialect-compatible vector of signless integer). Empty by default.
            no_wrap_flags: Property ``noWrapFlags`` (flags of ``GEPNoWrapFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def base(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``base``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def dynamic_indices(self) -> list[mlir_python._mlir_python.Value]:
        """
        Operand ``dynamicIndices``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def raw_constant_indices(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``rawConstantIndices``: i32 dense array attribute."""

    @raw_constant_indices.setter
    def raw_constant_indices(
        self, arg: mlir_python._mlir_python.Attribute, /
    ) -> None: ...
    @property
    def elem_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``elem_type``: any type attribute."""

    @elem_type.setter
    def elem_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def no_wrap_flags(self) -> GEPNoWrapFlags:
        """Property ``noWrapFlags``: flags of ``GEPNoWrapFlags``."""

    @no_wrap_flags.setter
    def no_wrap_flags(self, arg: GEPNoWrapFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.getelementptr"

class GlobalCtorsOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.global_ctors``: LLVM dialect global_ctors..

    Specifies a list of constructor functions, priorities, and associated data.
    The functions referenced by this array will be called in ascending order
    of priority (i.e. lowest first) when the module is loaded. The order of
    functions with the same priority is not defined. This operation is
    translated to LLVM's global_ctors global variable. The initializer
    functions are run at load time. However, if the associated data is not
    `#llvm.zero`, functions only run if the data is not discarded.

    Examples:

    ```mlir
    llvm.func @ctor() {
      ...
      llvm.return
    }
    llvm.mlir.global_ctors ctors = [@ctor], priorities = [0],
                                   data = [#llvm.zero]
    ```
    """

    def __init__(
        self,
        ctors: mlir_python._mlir_python.ArrayAttr,
        priorities: mlir_python._mlir_python.ArrayAttr,
        data: mlir_python._mlir_python.ArrayAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.global_ctors``: LLVM dialect global_ctors..

        Args:
            ctors: Attribute ``ctors`` (flat symbol ref array attribute).
            priorities: Attribute ``priorities`` (32-bit integer array attribute).
            data: Attribute ``data`` (array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def ctors(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``ctors``: flat symbol ref array attribute."""

    @ctors.setter
    def ctors(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def priorities(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``priorities``: 32-bit integer array attribute."""

    @priorities.setter
    def priorities(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def data(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``data``: array attribute."""

    @data.setter
    def data(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "llvm.mlir.global_ctors"

class GlobalDtorsOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.global_dtors``: LLVM dialect global_dtors..

    Specifies a list of destructor functions and priorities. The functions
    referenced by this array will be called in descending order of priority
    (i.e. highest first) when the module is unloaded. The order of functions
    with the same priority is not defined. This operation is translated to
    LLVM's global_dtors global variable. The destruction functions are run at
    load time. However, if the associated data is not `#llvm.zero`, functions
    only run if the data is not discarded.

    Examples:

    ```mlir
    llvm.func @dtor() {
      llvm.return
    }
    llvm.mlir.global_dtors dtors = [@dtor], priorities = [0],
                                   data = [#llvm.zero]
    ```
    """

    def __init__(
        self,
        dtors: mlir_python._mlir_python.ArrayAttr,
        priorities: mlir_python._mlir_python.ArrayAttr,
        data: mlir_python._mlir_python.ArrayAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.global_dtors``: LLVM dialect global_dtors..

        Args:
            dtors: Attribute ``dtors`` (flat symbol ref array attribute).
            priorities: Attribute ``priorities`` (32-bit integer array attribute).
            data: Attribute ``data`` (array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dtors(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``dtors``: flat symbol ref array attribute."""

    @dtors.setter
    def dtors(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def priorities(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``priorities``: 32-bit integer array attribute."""

    @priorities.setter
    def priorities(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def data(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``data``: array attribute."""

    @data.setter
    def data(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "llvm.mlir.global_dtors"

class GlobalOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.global``: LLVM dialect global..

    Since MLIR allows for arbitrary operations to be present at the top level,
    global variables are defined using the `llvm.mlir.global` operation. Both
    global constants and variables can be defined, and the value may also be
    initialized in both cases.

    There are two forms of initialization syntax. Simple constants that can be
    represented as MLIR attributes can be given in-line:

    ```mlir
    llvm.mlir.global @variable(32.0 : f32) : f32
    ```

    This initialization and type syntax is similar to `llvm.mlir.constant` and
    may use two types: one for MLIR attribute and another for the LLVM value.
    These types must be compatible.

    More complex constants that cannot be represented as MLIR attributes can be
    given in an initializer region:

    ```mlir
    // This global is initialized with the equivalent of:
    //   i32* getelementptr (i32* @g2, i32 2)
    llvm.mlir.global constant @int_gep() : !llvm.ptr {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr, i32) -> !llvm.ptr, i32
      // The initializer region must end with `llvm.return`.
      llvm.return %2 : !llvm.ptr
    }
    ```

    Only one of the initializer attribute or initializer region may be provided.

    `llvm.mlir.global` must appear at top-level of the enclosing module. It uses
    an @-identifier for its value, which will be uniqued by the module with
    respect to other @-identifiers in it.

    Examples:

    ```mlir
    // Global values use @-identifiers.
    llvm.mlir.global constant @cst(42 : i32) : i32

    // Non-constant values must also be initialized.
    llvm.mlir.global @variable(32.0 : f32) : f32

    // Strings are expected to be of wrapped LLVM i8 array type and do not
    // automatically include the trailing zero.
    llvm.mlir.global @string("abc") : !llvm.array<3 x i8>

    // For strings globals, the trailing type may be omitted.
    llvm.mlir.global constant @no_trailing_type("foo bar")

    // A complex initializer is constructed with an initializer region.
    llvm.mlir.global constant @int_gep() : !llvm.ptr {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr, i32) -> !llvm.ptr, i32
      llvm.return %2 : !llvm.ptr
    }
    ```

    Similarly to functions, globals have a linkage attribute. In the custom
    syntax, this attribute is placed between `llvm.mlir.global` and the optional
    `constant` keyword. If the attribute is omitted, `external` linkage is
    assumed by default.

    Examples:

    ```mlir
    // A constant with internal linkage will not participate in linking.
    llvm.mlir.global internal constant @cst(42 : i32) : i32

    // By default, "external" linkage is assumed and the global participates in
    // symbol resolution at link-time.
    llvm.mlir.global @glob(0 : f32) : f32

    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
    ```

    Like global variables in LLVM IR, globals can have an (optional)
    alignment attribute using keyword `alignment`. The integer value of the
    alignment must be a positive integer that is a power of 2.

    Examples:

    ```mlir
    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
    ```

    The `target_specific_attrs` attribute provides a mechanism to preserve
    target-specific LLVM IR attributes that are not explicitly modeled in the
    LLVM dialect.

    The attribute is an array containing either string attributes or
    two-element array attributes of strings. The value of a standalone string
    attribute is interpreted as the name of an LLVM IR attribute on the global.
    A two-element array is interpreted as a key-value pair.

    Example:

    ```mlir
    llvm.mlir.global external @example() {
      target_specific_attrs = ["value-less-attr", ["int-attr", "4"], ["string-attr", "string"]]} : f64
    ```
    """

    def __init__(
        self,
        global_type: mlir_python._mlir_python.Type,
        sym_name: str,
        linkage: Linkage,
        *,
        constant: bool = False,
        dso_local: bool = False,
        thread_local_: bool = False,
        externally_initialized: bool = False,
        value: mlir_python._mlir_python.Attribute | None = None,
        alignment: int | None = None,
        addr_space: int = 0,
        unnamed_addr: UnnamedAddr | None = None,
        section: str | None = None,
        comdat: mlir_python._mlir_python.SymbolRefAttr | None = None,
        dbg_exprs: mlir_python._mlir_python.ArrayAttr | None = None,
        visibility_: Visibility = Visibility.DEFAULT,
        target_specific_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.global``: LLVM dialect global..

        Args:
            global_type: Attribute ``global_type`` (any type attribute).
            sym_name: Attribute ``sym_name`` (string attribute).
            linkage: Attribute ``linkage`` (a member of ``Linkage``).
            constant: Attribute ``constant`` (unit attribute). Omit for the default.
            dso_local: Attribute ``dso_local`` (unit attribute). Omit for the default.
            thread_local_: Attribute ``thread_local_`` (unit attribute). Omit for the default.
            externally_initialized: Attribute ``externally_initialized`` (unit attribute). Omit for the default.
            value: Attribute ``value`` (any attribute). Optional.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            addr_space: Attribute ``addr_space`` (32-bit signless integer attribute whose value is non-negative). Omit for the default.
            unnamed_addr: Attribute ``unnamed_addr`` (a member of ``UnnamedAddr``). Optional.
            section: Attribute ``section`` (string attribute). Optional.
            comdat: Attribute ``comdat`` (symbol reference attribute). Optional.
            dbg_exprs: Attribute ``dbg_exprs`` (an array of variable expressions). Optional.
            visibility_: Attribute ``visibility_`` (a member of ``Visibility``). Omit for the default.
            target_specific_attrs: Attribute ``target_specific_attrs`` (array attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def global_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``global_type``: any type attribute."""

    @global_type.setter
    def global_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def constant(self) -> bool:
        """Attribute ``constant``: unit attribute."""

    @constant.setter
    def constant(self, arg: bool, /) -> None: ...
    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def linkage(self) -> Linkage:
        """Attribute ``linkage``: a member of ``Linkage``."""

    @linkage.setter
    def linkage(self, arg: Linkage, /) -> None: ...
    @property
    def dso_local(self) -> bool:
        """Attribute ``dso_local``: unit attribute."""

    @dso_local.setter
    def dso_local(self, arg: bool, /) -> None: ...
    @property
    def externally_initialized(self) -> bool:
        """Attribute ``externally_initialized``: unit attribute."""

    @externally_initialized.setter
    def externally_initialized(self, arg: bool, /) -> None: ...
    @property
    def value(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``value``: any attribute."""

    @value.setter
    def value(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def addr_space(self) -> int:
        """
        Attribute ``addr_space``: 32-bit signless integer attribute whose value is non-negative.
        """

    @addr_space.setter
    def addr_space(self, arg: int, /) -> None: ...
    @property
    def unnamed_addr(self) -> UnnamedAddr | None:
        """Attribute ``unnamed_addr``: a member of ``UnnamedAddr``."""

    @unnamed_addr.setter
    def unnamed_addr(self, arg: UnnamedAddr | None) -> None: ...
    @property
    def section(self) -> str | None:
        """Attribute ``section``: string attribute."""

    @section.setter
    def section(self, arg: str | None) -> None: ...
    @property
    def comdat(self) -> mlir_python._mlir_python.SymbolRefAttr | None:
        """Attribute ``comdat``: symbol reference attribute."""

    @comdat.setter
    def comdat(self, arg: mlir_python._mlir_python.SymbolRefAttr | None) -> None: ...
    @property
    def dbg_exprs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``dbg_exprs``: an array of variable expressions."""

    @dbg_exprs.setter
    def dbg_exprs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def target_specific_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``target_specific_attrs``: array attribute."""

    @target_specific_attrs.setter
    def target_specific_attrs(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def initializer(self) -> mlir_python._mlir_python.Region:
        """Region ``initializer``: any region."""

    OPERATION_NAME: str = "llvm.mlir.global"

class ICmpOp(mlir_python._mlir_python.Operation):
    """``llvm.icmp``."""

    def __init__(
        self,
        predicate: ICmpPredicate,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.icmp``.

        Result types are inferred.

        Args:
            predicate: Attribute ``predicate`` (a member of ``ICmpPredicate``).
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer.
        """

    @property
    def predicate(self) -> ICmpPredicate:
        """Attribute ``predicate``: a member of ``ICmpPredicate``."""

    @predicate.setter
    def predicate(self, arg: ICmpPredicate, /) -> None: ...

    OPERATION_NAME: str = "llvm.icmp"

class IFuncOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.ifunc``: LLVM dialect ifunc.

    `llvm.mlir.ifunc` is a top level operation that defines a global ifunc.
    It defines a new symbol and takes a symbol refering to a resolver function.
    IFuncs can be called as regular functions. The function type is the same
    as the IFuncType. The symbol is resolved at runtime by calling a resolver
    function.

    Examples:

    ```mlir
    // IFuncs resolve a symbol at runtime using a resovler function.
    llvm.mlir.ifunc external @foo: !llvm.func<f32 (i64)>, !llvm.ptr @resolver

    llvm.func @foo_1(i64) -> f32
    llvm.func @foo_2(i64) -> f32

    llvm.func @resolve_foo() -> !llvm.ptr attributes {
      %0 = llvm.mlir.addressof @foo_2 : !llvm.ptr
      %1 = llvm.mlir.addressof @foo_1 : !llvm.ptr

      // ... Logic selecting from foo_{1, 2}

      // Return function pointer to the selected function
      llvm.return %7 : !llvm.ptr
    }

    llvm.func @use_foo() {
      // IFuncs are called as regular functions
      %res = llvm.call @foo(%value) : i64 -> f32
    }
    ```
    """

    def __init__(
        self,
        sym_name: str,
        i_func_type: mlir_python._mlir_python.Type,
        resolver: str,
        resolver_type: mlir_python._mlir_python.Type,
        linkage: Linkage,
        *,
        dso_local: bool = False,
        address_space: int = 0,
        unnamed_addr: UnnamedAddr = UnnamedAddr.NONE,
        visibility_: Visibility = Visibility.DEFAULT,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.ifunc``: LLVM dialect ifunc.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            i_func_type: Attribute ``i_func_type`` (any type attribute).
            resolver: Attribute ``resolver`` (flat symbol reference attribute).
            resolver_type: Attribute ``resolver_type`` (any type attribute).
            linkage: Attribute ``linkage`` (a member of ``Linkage``).
            dso_local: Attribute ``dso_local`` (unit attribute). Omit for the default.
            address_space: Attribute ``address_space`` (32-bit signless integer attribute whose value is non-negative). Omit for the default.
            unnamed_addr: Attribute ``unnamed_addr`` (a member of ``UnnamedAddr``). Omit for the default.
            visibility_: Attribute ``visibility_`` (a member of ``Visibility``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def i_func_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``i_func_type``: any type attribute."""

    @i_func_type.setter
    def i_func_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def resolver(self) -> str:
        """Attribute ``resolver``: flat symbol reference attribute."""

    @resolver.setter
    def resolver(self, arg: str, /) -> None: ...
    @property
    def resolver_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``resolver_type``: any type attribute."""

    @resolver_type.setter
    def resolver_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def linkage(self) -> Linkage:
        """Attribute ``linkage``: a member of ``Linkage``."""

    @linkage.setter
    def linkage(self, arg: Linkage, /) -> None: ...
    @property
    def dso_local(self) -> bool:
        """Attribute ``dso_local``: unit attribute."""

    @dso_local.setter
    def dso_local(self, arg: bool, /) -> None: ...
    @property
    def address_space(self) -> int:
        """
        Attribute ``address_space``: 32-bit signless integer attribute whose value is non-negative.
        """

    @address_space.setter
    def address_space(self, arg: int, /) -> None: ...
    @property
    def unnamed_addr(self) -> UnnamedAddr:
        """Attribute ``unnamed_addr``: a member of ``UnnamedAddr``."""

    @unnamed_addr.setter
    def unnamed_addr(self, arg: UnnamedAddr, /) -> None: ...

    OPERATION_NAME: str = "llvm.mlir.ifunc"

class IndirectBrOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.indirectbr``.

    Transfer control flow to address in `$addr`. A list of possible target
    blocks in `$successors` can be provided and maybe used as a hint in LLVM:

    ```mlir
    ...
    llvm.func @g(...
      %dest = llvm.blockaddress <function = @g, tag = <id = 0>> : !llvm.ptr
      llvm.indirectbr %dest : !llvm.ptr, [
        ^head
      ]
    ^head:
      llvm.blocktag <id = 0>
      llvm.return %arg0 : i32
      ...
    ```

    It also supports a list of operands that can be passed to a target block:

    ```mlir
      llvm.indirectbr %dest : !llvm.ptr, [
        ^head(%arg0 : i32),
        ^tail(%arg1, %arg0 : i32, i32)
      ]
    ^head(%r0 : i32):
      llvm.return %r0 : i32
    ^tail(%r1 : i32, %r2 : i32):
      ...
    ```
    """

    def __init__(
        self,
        addr: mlir_python._mlir_python.Value,
        successors: Sequence[mlir_python._mlir_python.Block],
        succ_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.indirectbr``.

        Args:
            addr: Operand ``addr`` (LLVM pointer type).
            successors: Successor ``successors`` (any successor).
            succ_operands: Operand ``succOperands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def addr(self) -> mlir_python._mlir_python.Value:
        """Operand ``addr``: LLVM pointer type."""

    @property
    def succ_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``succOperands``: any type."""

    OPERATION_NAME: str = "llvm.indirectbr"

class InlineAsmOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.inline_asm``.

    The InlineAsmOp mirrors the underlying LLVM semantics with a notable
    exception: the embedded `asm_string` is not allowed to define or reference
    any symbol or any global variable: only the operands of the op may be read,
    written, or referenced.
    Attempting to define or reference any symbol or any global behavior is
    considered undefined behavior at this time.
    If `tail_call_kind` is used, the operation behaves like the specified
    tail call kind. The `musttail` kind it's not available for this operation,
    since it isn't supported by LLVM's inline asm.
    """

    def __init__(
        self,
        asm_string: str,
        constraints: str,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        has_side_effects: bool = False,
        is_align_stack: bool = False,
        tail_call_kind: mlir_python._mlir_python.Attribute | None = None,
        asm_dialect: AsmDialect | None = None,
        operand_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_type: mlir_python._mlir_python.Type | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.inline_asm``.

        Args:
            asm_string: Attribute ``asm_string`` (string attribute).
            constraints: Attribute ``constraints`` (string attribute).
            operands: Operand ``operands`` (LLVM dialect-compatible type). Empty by default.
            has_side_effects: Attribute ``has_side_effects`` (unit attribute). Omit for the default.
            is_align_stack: Attribute ``is_align_stack`` (unit attribute). Omit for the default.
            tail_call_kind: Attribute ``tail_call_kind`` (LLVM Calling Convention specification). Omit for the default.
            asm_dialect: Attribute ``asm_dialect`` (a member of ``AsmDialect``). Optional.
            operand_attrs: Attribute ``operand_attrs`` (array attribute). Optional.
            res_type: Type of result ``res`` (LLVM dialect-compatible type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult | None:
        """Result ``res``: LLVM dialect-compatible type."""

    @property
    def asm_string(self) -> str:
        """Attribute ``asm_string``: string attribute."""

    @asm_string.setter
    def asm_string(self, arg: str, /) -> None: ...
    @property
    def constraints(self) -> str:
        """Attribute ``constraints``: string attribute."""

    @constraints.setter
    def constraints(self, arg: str, /) -> None: ...
    @property
    def has_side_effects(self) -> bool:
        """Attribute ``has_side_effects``: unit attribute."""

    @has_side_effects.setter
    def has_side_effects(self, arg: bool, /) -> None: ...
    @property
    def is_align_stack(self) -> bool:
        """Attribute ``is_align_stack``: unit attribute."""

    @is_align_stack.setter
    def is_align_stack(self, arg: bool, /) -> None: ...
    @property
    def tail_call_kind(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``tail_call_kind``: LLVM Calling Convention specification."""

    @tail_call_kind.setter
    def tail_call_kind(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def asm_dialect(self) -> AsmDialect | None:
        """Attribute ``asm_dialect``: a member of ``AsmDialect``."""

    @asm_dialect.setter
    def asm_dialect(self, arg: AsmDialect | None) -> None: ...
    @property
    def operand_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``operand_attrs``: array attribute."""

    @operand_attrs.setter
    def operand_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.inline_asm"

class InsertElementOp(mlir_python._mlir_python.Operation):
    """``llvm.insertelement``: Insert an element into an LLVM vector.."""

    def __init__(
        self,
        vector: mlir_python._mlir_python.Value,
        value: mlir_python._mlir_python.Value,
        position: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.insertelement``: Insert an element into an LLVM vector..

        Result types are inferred.

        Args:
            vector: Operand ``vector`` (LLVM dialect-compatible vector type).
            value: Operand ``value`` (primitive LLVM type).
            position: Operand ``position`` (signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def vector(self) -> mlir_python._mlir_python.Value:
        """Operand ``vector``: LLVM dialect-compatible vector type."""

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: primitive LLVM type."""

    @property
    def position(self) -> mlir_python._mlir_python.Value:
        """Operand ``position``: signless integer."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible vector type."""

    OPERATION_NAME: str = "llvm.insertelement"

class InsertValueOp(mlir_python._mlir_python.Operation):
    """``llvm.insertvalue``: Insert a value into an LLVM struct.."""

    def __init__(
        self,
        container: mlir_python._mlir_python.Value,
        value: mlir_python._mlir_python.Value,
        position: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.insertvalue``: Insert a value into an LLVM struct..

        Result types are inferred.

        Args:
            container: Operand ``container`` (LLVM aggregate type).
            value: Operand ``value`` (primitive LLVM type).
            position: Attribute ``position`` (i64 dense array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def container(self) -> mlir_python._mlir_python.Value:
        """Operand ``container``: LLVM aggregate type."""

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: primitive LLVM type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM aggregate type."""

    @property
    def position(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``position``: i64 dense array attribute."""

    @position.setter
    def position(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.insertvalue"

class IntToPtrOp(mlir_python._mlir_python.Operation):
    """``llvm.inttoptr``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        dereferenceable: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.inttoptr``.

        Args:
            res_type: Type of result ``res`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            dereferenceable: Attribute ``dereferenceable`` (LLVM dereferenceable attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def dereferenceable(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``dereferenceable``: LLVM dereferenceable attribute."""

    @dereferenceable.setter
    def dereferenceable(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...

    OPERATION_NAME: str = "llvm.inttoptr"

class InvokeOp(mlir_python._mlir_python.Operation):
    """``llvm.invoke``."""

    def __init__(
        self,
        normal_dest: mlir_python._mlir_python.Block,
        unwind_dest: mlir_python._mlir_python.Block,
        callee_operands: Sequence[mlir_python._mlir_python.Value] = [],
        normal_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        unwind_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        op_bundle_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        var_callee_type: mlir_python._mlir_python.Type | None = None,
        callee: str | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        branch_weights: mlir_python._mlir_python.Attribute | None = None,
        cconv: mlir_python._mlir_python.Attribute | None = None,
        op_bundle_tags: mlir_python._mlir_python.ArrayAttr | None = None,
        result_type: mlir_python._mlir_python.Type | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.invoke``.

        Args:
            normal_dest: Successor ``normalDest`` (any successor).
            unwind_dest: Successor ``unwindDest`` (any successor).
            callee_operands: Operand ``callee_operands`` (LLVM dialect-compatible type). Empty by default.
            normal_dest_operands: Operand ``normalDestOperands`` (LLVM dialect-compatible type). Empty by default.
            unwind_dest_operands: Operand ``unwindDestOperands`` (LLVM dialect-compatible type). Empty by default.
            op_bundle_operands: Operand ``op_bundle_operands`` (LLVM dialect-compatible type). Empty by default.
            var_callee_type: Attribute ``var_callee_type`` (type attribute of LLVM function type). Optional.
            callee: Attribute ``callee`` (flat symbol reference attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            branch_weights: Attribute ``branch_weights`` (i32 dense array attribute). Optional.
            cconv: Attribute ``CConv`` (LLVM Calling Convention specification). Omit for the default.
            op_bundle_tags: Attribute ``op_bundle_tags`` (array attribute). Optional.
            result_type: Type of result ``result`` (LLVM dialect-compatible type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``callee_operands``: LLVM dialect-compatible type."""

    @property
    def normal_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``normalDestOperands``: LLVM dialect-compatible type."""

    @property
    def unwind_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``unwindDestOperands``: LLVM dialect-compatible type."""

    @property
    def op_bundle_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``op_bundle_operands``: LLVM dialect-compatible type."""

    @property
    def var_callee_type(self) -> mlir_python._mlir_python.Type | None:
        """Attribute ``var_callee_type``: type attribute of LLVM function type."""

    @var_callee_type.setter
    def var_callee_type(self, arg: mlir_python._mlir_python.Type | None) -> None: ...
    @property
    def callee(self) -> str | None:
        """Attribute ``callee``: flat symbol reference attribute."""

    @callee.setter
    def callee(self, arg: str | None) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def branch_weights(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``branch_weights``: i32 dense array attribute."""

    @branch_weights.setter
    def branch_weights(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def cconv(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``CConv``: LLVM Calling Convention specification."""

    @cconv.setter
    def cconv(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def op_bundle_tags(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``op_bundle_tags``: array attribute."""

    @op_bundle_tags.setter
    def op_bundle_tags(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def normal_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``normalDest``."""

    @property
    def unwind_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``unwindDest``."""

    OPERATION_NAME: str = "llvm.invoke"

class LLVMFuncOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.func``: LLVM dialect function..

    MLIR functions are defined by an operation that is not built into the IR
    itself. The LLVM dialect provides an `llvm.func` operation to define
    functions compatible with LLVM IR. These functions have LLVM dialect
    function type but use MLIR syntax to express it. They are required to have
    exactly one result type. LLVM function operation is intended to capture
    additional properties of LLVM functions, such as linkage and calling
    convention, that may be modeled differently by the built-in MLIR function.

    ```mlir
    // The type of @bar is !llvm<"i64 (i64)">
    llvm.func @bar(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
    }

    // Type type of @foo is !llvm<"void (i64)">
    // !llvm.void type is omitted
    llvm.func @foo(%arg0: i64) {
      llvm.return
    }

    // A function with `internal` linkage.
    llvm.func internal @internal_func() {
      llvm.return
    }
    ```
    """

    def __init__(
        self,
        sym_name: str,
        function_type: mlir_python._mlir_python.Type,
        *,
        sym_visibility: str | None = None,
        linkage: Linkage = Linkage.EXTERNAL,
        dso_local: bool = False,
        cconv: mlir_python._mlir_python.Attribute | None = None,
        comdat: mlir_python._mlir_python.SymbolRefAttr | None = None,
        convergent: bool | None = None,
        personality: str | None = None,
        garbage_collector: str | None = None,
        passthrough: mlir_python._mlir_python.ArrayAttr | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        function_entry_count: int | None = None,
        memory_effects: mlir_python._mlir_python.Attribute | None = None,
        visibility_: Visibility = Visibility.DEFAULT,
        arm_streaming: bool | None = None,
        arm_locally_streaming: bool | None = None,
        arm_streaming_compatible: bool | None = None,
        arm_new_za: bool | None = None,
        arm_in_za: bool | None = None,
        arm_out_za: bool | None = None,
        arm_inout_za: bool | None = None,
        arm_preserves_za: bool | None = None,
        section: str | None = None,
        unnamed_addr: UnnamedAddr | None = None,
        alignment: int | None = None,
        vscale_range: mlir_python._mlir_python.Attribute | None = None,
        frame_pointer: FramePointerKind | None = None,
        target_cpu: str | None = None,
        tune_cpu: str | None = None,
        reciprocal_estimates: str | None = None,
        prefer_vector_width: str | None = None,
        target_features: mlir_python._mlir_python.Attribute | None = None,
        no_infs_fp_math: bool | None = None,
        no_nans_fp_math: bool | None = None,
        no_signed_zeros_fp_math: bool | None = None,
        denormal_fp_math: str | None = None,
        denormal_fp_math_f32: str | None = None,
        fp_contract: str | None = None,
        instrument_function_entry: str | None = None,
        instrument_function_exit: str | None = None,
        no_inline: bool | None = None,
        always_inline: bool | None = None,
        inline_hint: bool | None = None,
        no_unwind: bool | None = None,
        will_return: bool | None = None,
        optimize_none: bool | None = None,
        vec_type_hint: mlir_python._mlir_python.Attribute | None = None,
        work_group_size_hint: mlir_python._mlir_python.Attribute | None = None,
        reqd_work_group_size: mlir_python._mlir_python.Attribute | None = None,
        intel_reqd_sub_group_size: int | None = None,
        uwtable_kind: UWTableKind | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.func``: LLVM dialect function..

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            function_type: Attribute ``function_type`` (type attribute of LLVM function type).
            sym_visibility: Attribute ``sym_visibility`` (string attribute). Optional.
            linkage: Attribute ``linkage`` (a member of ``Linkage``). Omit for the default.
            dso_local: Attribute ``dso_local`` (unit attribute). Omit for the default.
            cconv: Attribute ``CConv`` (LLVM Calling Convention specification). Omit for the default.
            comdat: Attribute ``comdat`` (symbol reference attribute). Optional.
            convergent: Attribute ``convergent`` (unit attribute). Optional.
            personality: Attribute ``personality`` (flat symbol reference attribute). Optional.
            garbage_collector: Attribute ``garbageCollector`` (string attribute). Optional.
            passthrough: Attribute ``passthrough`` (array attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            function_entry_count: Attribute ``function_entry_count`` (64-bit signless integer attribute). Optional.
            memory_effects: Attribute ``memory_effects``. Optional.
            visibility_: Attribute ``visibility_`` (a member of ``Visibility``). Omit for the default.
            arm_streaming: Attribute ``arm_streaming`` (unit attribute). Optional.
            arm_locally_streaming: Attribute ``arm_locally_streaming`` (unit attribute). Optional.
            arm_streaming_compatible: Attribute ``arm_streaming_compatible`` (unit attribute). Optional.
            arm_new_za: Attribute ``arm_new_za`` (unit attribute). Optional.
            arm_in_za: Attribute ``arm_in_za`` (unit attribute). Optional.
            arm_out_za: Attribute ``arm_out_za`` (unit attribute). Optional.
            arm_inout_za: Attribute ``arm_inout_za`` (unit attribute). Optional.
            arm_preserves_za: Attribute ``arm_preserves_za`` (unit attribute). Optional.
            section: Attribute ``section`` (string attribute). Optional.
            unnamed_addr: Attribute ``unnamed_addr`` (a member of ``UnnamedAddr``). Optional.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            vscale_range: Attribute ``vscale_range``. Optional.
            frame_pointer: Attribute ``frame_pointer`` (a member of ``FramePointerKind``). Optional.
            target_cpu: Attribute ``target_cpu`` (string attribute). Optional.
            tune_cpu: Attribute ``tune_cpu`` (string attribute). Optional.
            reciprocal_estimates: Attribute ``reciprocal_estimates`` (string attribute). Optional.
            prefer_vector_width: Attribute ``prefer_vector_width`` (string attribute). Optional.
            target_features: Attribute ``target_features`` (LLVM target features attribute). Optional.
            no_infs_fp_math: Attribute ``no_infs_fp_math`` (bool attribute). Optional.
            no_nans_fp_math: Attribute ``no_nans_fp_math`` (bool attribute). Optional.
            no_signed_zeros_fp_math: Attribute ``no_signed_zeros_fp_math`` (bool attribute). Optional.
            denormal_fp_math: Attribute ``denormal_fp_math`` (string attribute). Optional.
            denormal_fp_math_f32: Attribute ``denormal_fp_math_f32`` (string attribute). Optional.
            fp_contract: Attribute ``fp_contract`` (string attribute). Optional.
            instrument_function_entry: Attribute ``instrument_function_entry`` (string attribute). Optional.
            instrument_function_exit: Attribute ``instrument_function_exit`` (string attribute). Optional.
            no_inline: Attribute ``no_inline`` (unit attribute). Optional.
            always_inline: Attribute ``always_inline`` (unit attribute). Optional.
            inline_hint: Attribute ``inline_hint`` (unit attribute). Optional.
            no_unwind: Attribute ``no_unwind`` (unit attribute). Optional.
            will_return: Attribute ``will_return`` (unit attribute). Optional.
            optimize_none: Attribute ``optimize_none`` (unit attribute). Optional.
            vec_type_hint: Attribute ``vec_type_hint`` (Explicit vectorization compiler hint). Optional.
            work_group_size_hint: Attribute ``work_group_size_hint`` (i32 dense array attribute). Optional.
            reqd_work_group_size: Attribute ``reqd_work_group_size`` (i32 dense array attribute). Optional.
            intel_reqd_sub_group_size: Attribute ``intel_reqd_sub_group_size`` (32-bit signless integer attribute). Optional.
            uwtable_kind: Attribute ``uwtable_kind`` (a member of ``UWTableKind``). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def sym_visibility(self) -> str | None:
        """Attribute ``sym_visibility``: string attribute."""

    @sym_visibility.setter
    def sym_visibility(self, arg: str | None) -> None: ...
    @property
    def function_type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``function_type``: type attribute of LLVM function type."""

    @function_type.setter
    def function_type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def linkage(self) -> Linkage:
        """Attribute ``linkage``: a member of ``Linkage``."""

    @linkage.setter
    def linkage(self, arg: Linkage, /) -> None: ...
    @property
    def dso_local(self) -> bool:
        """Attribute ``dso_local``: unit attribute."""

    @dso_local.setter
    def dso_local(self, arg: bool, /) -> None: ...
    @property
    def cconv(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``CConv``: LLVM Calling Convention specification."""

    @cconv.setter
    def cconv(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def comdat(self) -> mlir_python._mlir_python.SymbolRefAttr | None:
        """Attribute ``comdat``: symbol reference attribute."""

    @comdat.setter
    def comdat(self, arg: mlir_python._mlir_python.SymbolRefAttr | None) -> None: ...
    @property
    def convergent(self) -> bool | None:
        """Attribute ``convergent``: unit attribute."""

    @convergent.setter
    def convergent(self, arg: bool | None) -> None: ...
    @property
    def personality(self) -> str | None:
        """Attribute ``personality``: flat symbol reference attribute."""

    @personality.setter
    def personality(self, arg: str | None) -> None: ...
    @property
    def garbage_collector(self) -> str | None:
        """Attribute ``garbageCollector``: string attribute."""

    @garbage_collector.setter
    def garbage_collector(self, arg: str | None) -> None: ...
    @property
    def passthrough(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``passthrough``: array attribute."""

    @passthrough.setter
    def passthrough(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def function_entry_count(self) -> int | None:
        """Attribute ``function_entry_count``: 64-bit signless integer attribute."""

    @function_entry_count.setter
    def function_entry_count(self, arg: int | None) -> None: ...
    @property
    def memory_effects(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``memory_effects``."""

    @memory_effects.setter
    def memory_effects(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def arm_streaming(self) -> bool | None:
        """Attribute ``arm_streaming``: unit attribute."""

    @arm_streaming.setter
    def arm_streaming(self, arg: bool | None) -> None: ...
    @property
    def arm_locally_streaming(self) -> bool | None:
        """Attribute ``arm_locally_streaming``: unit attribute."""

    @arm_locally_streaming.setter
    def arm_locally_streaming(self, arg: bool | None) -> None: ...
    @property
    def arm_streaming_compatible(self) -> bool | None:
        """Attribute ``arm_streaming_compatible``: unit attribute."""

    @arm_streaming_compatible.setter
    def arm_streaming_compatible(self, arg: bool | None) -> None: ...
    @property
    def arm_new_za(self) -> bool | None:
        """Attribute ``arm_new_za``: unit attribute."""

    @arm_new_za.setter
    def arm_new_za(self, arg: bool | None) -> None: ...
    @property
    def arm_in_za(self) -> bool | None:
        """Attribute ``arm_in_za``: unit attribute."""

    @arm_in_za.setter
    def arm_in_za(self, arg: bool | None) -> None: ...
    @property
    def arm_out_za(self) -> bool | None:
        """Attribute ``arm_out_za``: unit attribute."""

    @arm_out_za.setter
    def arm_out_za(self, arg: bool | None) -> None: ...
    @property
    def arm_inout_za(self) -> bool | None:
        """Attribute ``arm_inout_za``: unit attribute."""

    @arm_inout_za.setter
    def arm_inout_za(self, arg: bool | None) -> None: ...
    @property
    def arm_preserves_za(self) -> bool | None:
        """Attribute ``arm_preserves_za``: unit attribute."""

    @arm_preserves_za.setter
    def arm_preserves_za(self, arg: bool | None) -> None: ...
    @property
    def section(self) -> str | None:
        """Attribute ``section``: string attribute."""

    @section.setter
    def section(self, arg: str | None) -> None: ...
    @property
    def unnamed_addr(self) -> UnnamedAddr | None:
        """Attribute ``unnamed_addr``: a member of ``UnnamedAddr``."""

    @unnamed_addr.setter
    def unnamed_addr(self, arg: UnnamedAddr | None) -> None: ...
    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def vscale_range(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``vscale_range``."""

    @vscale_range.setter
    def vscale_range(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def frame_pointer(self) -> FramePointerKind | None:
        """Attribute ``frame_pointer``: a member of ``FramePointerKind``."""

    @frame_pointer.setter
    def frame_pointer(self, arg: FramePointerKind | None) -> None: ...
    @property
    def target_cpu(self) -> str | None:
        """Attribute ``target_cpu``: string attribute."""

    @target_cpu.setter
    def target_cpu(self, arg: str | None) -> None: ...
    @property
    def tune_cpu(self) -> str | None:
        """Attribute ``tune_cpu``: string attribute."""

    @tune_cpu.setter
    def tune_cpu(self, arg: str | None) -> None: ...
    @property
    def reciprocal_estimates(self) -> str | None:
        """Attribute ``reciprocal_estimates``: string attribute."""

    @reciprocal_estimates.setter
    def reciprocal_estimates(self, arg: str | None) -> None: ...
    @property
    def prefer_vector_width(self) -> str | None:
        """Attribute ``prefer_vector_width``: string attribute."""

    @prefer_vector_width.setter
    def prefer_vector_width(self, arg: str | None) -> None: ...
    @property
    def target_features(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``target_features``: LLVM target features attribute."""

    @target_features.setter
    def target_features(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def no_infs_fp_math(self) -> bool | None:
        """Attribute ``no_infs_fp_math``: bool attribute."""

    @no_infs_fp_math.setter
    def no_infs_fp_math(self, arg: bool | None) -> None: ...
    @property
    def no_nans_fp_math(self) -> bool | None:
        """Attribute ``no_nans_fp_math``: bool attribute."""

    @no_nans_fp_math.setter
    def no_nans_fp_math(self, arg: bool | None) -> None: ...
    @property
    def no_signed_zeros_fp_math(self) -> bool | None:
        """Attribute ``no_signed_zeros_fp_math``: bool attribute."""

    @no_signed_zeros_fp_math.setter
    def no_signed_zeros_fp_math(self, arg: bool | None) -> None: ...
    @property
    def denormal_fp_math(self) -> str | None:
        """Attribute ``denormal_fp_math``: string attribute."""

    @denormal_fp_math.setter
    def denormal_fp_math(self, arg: str | None) -> None: ...
    @property
    def denormal_fp_math_f32(self) -> str | None:
        """Attribute ``denormal_fp_math_f32``: string attribute."""

    @denormal_fp_math_f32.setter
    def denormal_fp_math_f32(self, arg: str | None) -> None: ...
    @property
    def fp_contract(self) -> str | None:
        """Attribute ``fp_contract``: string attribute."""

    @fp_contract.setter
    def fp_contract(self, arg: str | None) -> None: ...
    @property
    def instrument_function_entry(self) -> str | None:
        """Attribute ``instrument_function_entry``: string attribute."""

    @instrument_function_entry.setter
    def instrument_function_entry(self, arg: str | None) -> None: ...
    @property
    def instrument_function_exit(self) -> str | None:
        """Attribute ``instrument_function_exit``: string attribute."""

    @instrument_function_exit.setter
    def instrument_function_exit(self, arg: str | None) -> None: ...
    @property
    def no_inline(self) -> bool | None:
        """Attribute ``no_inline``: unit attribute."""

    @no_inline.setter
    def no_inline(self, arg: bool | None) -> None: ...
    @property
    def always_inline(self) -> bool | None:
        """Attribute ``always_inline``: unit attribute."""

    @always_inline.setter
    def always_inline(self, arg: bool | None) -> None: ...
    @property
    def inline_hint(self) -> bool | None:
        """Attribute ``inline_hint``: unit attribute."""

    @inline_hint.setter
    def inline_hint(self, arg: bool | None) -> None: ...
    @property
    def no_unwind(self) -> bool | None:
        """Attribute ``no_unwind``: unit attribute."""

    @no_unwind.setter
    def no_unwind(self, arg: bool | None) -> None: ...
    @property
    def will_return(self) -> bool | None:
        """Attribute ``will_return``: unit attribute."""

    @will_return.setter
    def will_return(self, arg: bool | None) -> None: ...
    @property
    def optimize_none(self) -> bool | None:
        """Attribute ``optimize_none``: unit attribute."""

    @optimize_none.setter
    def optimize_none(self, arg: bool | None) -> None: ...
    @property
    def vec_type_hint(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``vec_type_hint``: Explicit vectorization compiler hint."""

    @vec_type_hint.setter
    def vec_type_hint(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def work_group_size_hint(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``work_group_size_hint``: i32 dense array attribute."""

    @work_group_size_hint.setter
    def work_group_size_hint(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def reqd_work_group_size(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``reqd_work_group_size``: i32 dense array attribute."""

    @reqd_work_group_size.setter
    def reqd_work_group_size(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def intel_reqd_sub_group_size(self) -> int | None:
        """
        Attribute ``intel_reqd_sub_group_size``: 32-bit signless integer attribute.
        """

    @intel_reqd_sub_group_size.setter
    def intel_reqd_sub_group_size(self, arg: int | None) -> None: ...
    @property
    def uwtable_kind(self) -> UWTableKind | None:
        """Attribute ``uwtable_kind``: a member of ``UWTableKind``."""

    @uwtable_kind.setter
    def uwtable_kind(self, arg: UWTableKind | None) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: any region."""

    OPERATION_NAME: str = "llvm.func"

class LShrOp(mlir_python._mlir_python.Operation):
    """``llvm.lshr``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.lshr``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.lshr"

class LandingpadOp(mlir_python._mlir_python.Operation):
    """``llvm.landingpad``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        operand_1: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        cleanup: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.landingpad``.

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            operand_1: Operand ``operand_1`` (LLVM dialect-compatible type). Empty by default.
            cleanup: Attribute ``cleanup`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_1(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``operand_1``: LLVM dialect-compatible type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    @property
    def cleanup(self) -> bool:
        """Attribute ``cleanup``: unit attribute."""

    @cleanup.setter
    def cleanup(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.landingpad"

class LinkerOptionsOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.linker_options``: Options to pass to the linker when the object file is linked.

    Pass the given options to the linker when the resulting object file is linked.
    This is used extensively on Windows to determine the C runtime that the object
    files should link against.

    Examples:
    ```mlir
    // Link against the MSVC static threaded CRT.
    llvm.linker_options ["/DEFAULTLIB:", "libcmt"]

    // Link against aarch64 compiler-rt builtins
    llvm.linker_options ["-l", "clang_rt.builtins-aarch64"]
    ```
    """

    def __init__(
        self,
        options: mlir_python._mlir_python.ArrayAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.linker_options``: Options to pass to the linker when the object file is linked.

        Args:
            options: Attribute ``options`` (string array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def options(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``options``: string array attribute."""

    @options.setter
    def options(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "llvm.linker_options"

class LoadOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.load``.

    The `load` operation is used to read from memory. A load may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic load only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile load of a float variable.
    %0 = llvm.load volatile %ptr : !llvm.ptr -> f32

    // A nontemporal load of a float variable.
    %0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

    // An atomic load of an integer variable.
    %0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
        : !llvm.ptr -> i64
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#load-instruction
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        addr: mlir_python._mlir_python.Value,
        *,
        alignment: int | None = None,
        volatile_: bool = False,
        nontemporal: bool = False,
        invariant: bool = False,
        invariant_group: bool = False,
        ordering: AtomicOrdering = AtomicOrdering.NOT_ATOMIC,
        syncscope: str | None = None,
        dereferenceable: mlir_python._mlir_python.Attribute | None = None,
        access_groups: mlir_python._mlir_python.ArrayAttr | None = None,
        alias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        noalias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        tbaa: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.load``.

        Args:
            res_type: Type of result ``res`` (LLVM type with size).
            addr: Operand ``addr`` (LLVM pointer type).
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            volatile_: Attribute ``volatile_`` (unit attribute). Omit for the default.
            nontemporal: Attribute ``nontemporal`` (unit attribute). Omit for the default.
            invariant: Attribute ``invariant`` (unit attribute). Omit for the default.
            invariant_group: Attribute ``invariantGroup`` (unit attribute). Omit for the default.
            ordering: Attribute ``ordering`` (a member of ``AtomicOrdering``). Omit for the default.
            syncscope: Attribute ``syncscope`` (string attribute). Optional.
            dereferenceable: Attribute ``dereferenceable`` (LLVM dereferenceable attribute). Optional.
            access_groups: Attribute ``access_groups`` (LLVM dialect access group metadata array). Optional.
            alias_scopes: Attribute ``alias_scopes`` (LLVM dialect alias scope array). Optional.
            noalias_scopes: Attribute ``noalias_scopes`` (LLVM dialect alias scope array). Optional.
            tbaa: Attribute ``tbaa`` (LLVM dialect TBAA tag metadata array). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def addr(self) -> mlir_python._mlir_python.Value:
        """Operand ``addr``: LLVM pointer type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM type with size."""

    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def nontemporal(self) -> bool:
        """Attribute ``nontemporal``: unit attribute."""

    @nontemporal.setter
    def nontemporal(self, arg: bool, /) -> None: ...
    @property
    def invariant(self) -> bool:
        """Attribute ``invariant``: unit attribute."""

    @invariant.setter
    def invariant(self, arg: bool, /) -> None: ...
    @property
    def invariant_group(self) -> bool:
        """Attribute ``invariantGroup``: unit attribute."""

    @invariant_group.setter
    def invariant_group(self, arg: bool, /) -> None: ...
    @property
    def ordering(self) -> AtomicOrdering:
        """Attribute ``ordering``: a member of ``AtomicOrdering``."""

    @ordering.setter
    def ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def syncscope(self) -> str | None:
        """Attribute ``syncscope``: string attribute."""

    @syncscope.setter
    def syncscope(self, arg: str | None) -> None: ...
    @property
    def dereferenceable(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``dereferenceable``: LLVM dereferenceable attribute."""

    @dereferenceable.setter
    def dereferenceable(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def access_groups(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``access_groups``: LLVM dialect access group metadata array."""

    @access_groups.setter
    def access_groups(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def alias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``alias_scopes``: LLVM dialect alias scope array."""

    @alias_scopes.setter
    def alias_scopes(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def noalias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``noalias_scopes``: LLVM dialect alias scope array."""

    @noalias_scopes.setter
    def noalias_scopes(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def tbaa(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``tbaa``: LLVM dialect TBAA tag metadata array."""

    @tbaa.setter
    def tbaa(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.load"

class ModuleFlagsOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.module_flags``: Information about module properties.

    Represents the equivalent in MLIR for LLVM's `llvm.module.flags` metadata,
    which requires a list of metadata triplets. Each triplet entry is described
    by a `ModuleFlagAttr`.

    Example:
    ```mlir
    llvm.module.flags [
      #llvm.mlir.module_flag<error, "wchar_size", 4>,
      #llvm.mlir.module_flag<max, "PIC Level", 2>
    ]
    ```
    """

    def __init__(
        self,
        flags: mlir_python._mlir_python.ArrayAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.module_flags``: Information about module properties.

        Args:
            flags: Attribute ``flags`` (array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def flags(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``flags``: array attribute."""

    @flags.setter
    def flags(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "llvm.module_flags"

class MulOp(mlir_python._mlir_python.Operation):
    """``llvm.mul``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mul``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            overflow_flags: Property ``overflowFlags`` (flags of ``IntegerOverflowFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Property ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.mul"

class NoneTokenOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.none``: Defines a value containing an empty token to LLVM type..

    Unlike LLVM IR, MLIR does not have first-class token values. They must be
    explicitly created as SSA values using `llvm.mlir.none`. This operation has
    no operands or attributes, and returns a none token value of a wrapped LLVM IR
    pointer type.

    Examples:

    ```mlir
    %0 = llvm.mlir.none : !llvm.token
    ```
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.none``: Defines a value containing an empty token to LLVM type..

        Result types are inferred.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM token type."""

    OPERATION_NAME: str = "llvm.mlir.none"

class OrOp(mlir_python._mlir_python.Operation):
    """``llvm.or``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_disjoint: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.or``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            is_disjoint: Attribute ``isDisjoint`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def is_disjoint(self) -> bool:
        """Attribute ``isDisjoint``: unit attribute."""

    @is_disjoint.setter
    def is_disjoint(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.or"

class PoisonOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.poison``: Creates a poison value of LLVM dialect type..

    Unlike LLVM IR, MLIR does not have first-class poison values. Such values
    must be created as SSA values using `llvm.mlir.poison`. This operation has
    no operands or attributes. It creates a poison value of the specified LLVM
    IR dialect type.

    Example:

    ```mlir
    // Create a poison value for a structure with a 32-bit integer followed
    // by a float.
    %0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.poison``: Creates a poison value of LLVM dialect type..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.mlir.poison"

class PtrToIntOp(mlir_python._mlir_python.Operation):
    """``llvm.ptrtoint``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.ptrtoint``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.ptrtoint"

class ResumeOp(mlir_python._mlir_python.Operation):
    """``llvm.resume``."""

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.resume``.

        Args:
            value: Operand ``value`` (LLVM dialect-compatible type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.resume"

class ReturnOp(mlir_python._mlir_python.Operation):
    """``llvm.return``."""

    def __init__(
        self,
        *,
        arg: mlir_python._mlir_python.Value | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.return``.

        Args:
            arg: Operand ``arg`` (LLVM dialect-compatible type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value | None:
        """Operand ``arg``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.return"

class SDivOp(mlir_python._mlir_python.Operation):
    """``llvm.sdiv``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.sdiv``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.sdiv"

class SExtOp(mlir_python._mlir_python.Operation):
    """``llvm.sext``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.sext``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.sext"

class SIToFPOp(mlir_python._mlir_python.Operation):
    """``llvm.sitofp``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.sitofp``.

        Args:
            res_type: Type of result ``res`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    OPERATION_NAME: str = "llvm.sitofp"

class SRemOp(mlir_python._mlir_python.Operation):
    """``llvm.srem``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.srem``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.srem"

class SelectOp(mlir_python._mlir_python.Operation):
    """``llvm.select``."""

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        true_value: mlir_python._mlir_python.Value,
        false_value: mlir_python._mlir_python.Value,
        *,
        fastmath_flags: FastmathFlags = FastmathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.select``.

        Result types are inferred.

        Args:
            condition: Operand ``condition`` (1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer).
            true_value: Operand ``trueValue`` (LLVM dialect-compatible type).
            false_value: Operand ``falseValue`` (LLVM dialect-compatible type).
            fastmath_flags: Attribute ``fastmathFlags`` (flags of ``FastmathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``condition``: 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer.
        """

    @property
    def true_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``trueValue``: LLVM dialect-compatible type."""

    @property
    def false_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``falseValue``: LLVM dialect-compatible type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    @property
    def fastmath_flags(self) -> FastmathFlags:
        """Attribute ``fastmathFlags``: flags of ``FastmathFlags``."""

    @fastmath_flags.setter
    def fastmath_flags(self, arg: FastmathFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.select"

class ShlOp(mlir_python._mlir_python.Operation):
    """``llvm.shl``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.shl``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            overflow_flags: Property ``overflowFlags`` (flags of ``IntegerOverflowFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Property ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.shl"

class ShuffleVectorOp(mlir_python._mlir_python.Operation):
    """``llvm.shufflevector``: Construct a permutation of two vectors.."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        v1: mlir_python._mlir_python.Value,
        v2: mlir_python._mlir_python.Value,
        mask: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.shufflevector``: Construct a permutation of two vectors..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible vector type).
            v1: Operand ``v1`` (LLVM dialect-compatible vector type).
            v2: Operand ``v2`` (LLVM dialect-compatible vector type).
            mask: Attribute ``mask`` (i32 dense array attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def v1(self) -> mlir_python._mlir_python.Value:
        """Operand ``v1``: LLVM dialect-compatible vector type."""

    @property
    def v2(self) -> mlir_python._mlir_python.Value:
        """Operand ``v2``: LLVM dialect-compatible vector type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible vector type."""

    @property
    def mask(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``mask``: i32 dense array attribute."""

    @mask.setter
    def mask(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "llvm.shufflevector"

class StoreOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.store``.

    The `store` operation is used to write to memory. A store may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic store only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile store of a float variable.
    llvm.store volatile %val, %ptr : f32, !llvm.ptr

    // A nontemporal store of a float variable.
    llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

    // An atomic store of an integer variable.
    llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
        : i64, !llvm.ptr
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#store-instruction
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        addr: mlir_python._mlir_python.Value,
        *,
        alignment: int | None = None,
        volatile_: bool = False,
        nontemporal: bool = False,
        invariant_group: bool = False,
        ordering: AtomicOrdering = AtomicOrdering.NOT_ATOMIC,
        syncscope: str | None = None,
        access_groups: mlir_python._mlir_python.ArrayAttr | None = None,
        alias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        noalias_scopes: mlir_python._mlir_python.ArrayAttr | None = None,
        tbaa: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.store``.

        Args:
            value: Operand ``value`` (LLVM type with size).
            addr: Operand ``addr`` (LLVM pointer type).
            alignment: Attribute ``alignment`` (64-bit signless integer attribute). Optional.
            volatile_: Attribute ``volatile_`` (unit attribute). Omit for the default.
            nontemporal: Attribute ``nontemporal`` (unit attribute). Omit for the default.
            invariant_group: Attribute ``invariantGroup`` (unit attribute). Omit for the default.
            ordering: Attribute ``ordering`` (a member of ``AtomicOrdering``). Omit for the default.
            syncscope: Attribute ``syncscope`` (string attribute). Optional.
            access_groups: Attribute ``access_groups`` (LLVM dialect access group metadata array). Optional.
            alias_scopes: Attribute ``alias_scopes`` (LLVM dialect alias scope array). Optional.
            noalias_scopes: Attribute ``noalias_scopes`` (LLVM dialect alias scope array). Optional.
            tbaa: Attribute ``tbaa`` (LLVM dialect TBAA tag metadata array). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: LLVM type with size."""

    @property
    def addr(self) -> mlir_python._mlir_python.Value:
        """Operand ``addr``: LLVM pointer type."""

    @property
    def alignment(self) -> int | None:
        """Attribute ``alignment``: 64-bit signless integer attribute."""

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...
    @property
    def nontemporal(self) -> bool:
        """Attribute ``nontemporal``: unit attribute."""

    @nontemporal.setter
    def nontemporal(self, arg: bool, /) -> None: ...
    @property
    def invariant_group(self) -> bool:
        """Attribute ``invariantGroup``: unit attribute."""

    @invariant_group.setter
    def invariant_group(self, arg: bool, /) -> None: ...
    @property
    def ordering(self) -> AtomicOrdering:
        """Attribute ``ordering``: a member of ``AtomicOrdering``."""

    @ordering.setter
    def ordering(self, arg: AtomicOrdering, /) -> None: ...
    @property
    def syncscope(self) -> str | None:
        """Attribute ``syncscope``: string attribute."""

    @syncscope.setter
    def syncscope(self, arg: str | None) -> None: ...
    @property
    def access_groups(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``access_groups``: LLVM dialect access group metadata array."""

    @access_groups.setter
    def access_groups(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def alias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``alias_scopes``: LLVM dialect alias scope array."""

    @alias_scopes.setter
    def alias_scopes(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def noalias_scopes(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``noalias_scopes``: LLVM dialect alias scope array."""

    @noalias_scopes.setter
    def noalias_scopes(
        self, arg: mlir_python._mlir_python.ArrayAttr | None
    ) -> None: ...
    @property
    def tbaa(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``tbaa``: LLVM dialect TBAA tag metadata array."""

    @tbaa.setter
    def tbaa(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "llvm.store"

class SubOp(mlir_python._mlir_python.Operation):
    """``llvm.sub``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.sub``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            overflow_flags: Property ``overflowFlags`` (flags of ``IntegerOverflowFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Property ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.sub"

class SwitchOp(mlir_python._mlir_python.Operation):
    """``llvm.switch``."""

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        default_destination: mlir_python._mlir_python.Block,
        case_destinations: Sequence[mlir_python._mlir_python.Block],
        default_operands: Sequence[mlir_python._mlir_python.Value] = [],
        case_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        case_values: mlir_python._mlir_python.DenseIntElementsAttr | None = None,
        branch_weights: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.switch``.

        Args:
            value: Operand ``value`` (signless integer).
            default_destination: Successor ``defaultDestination`` (any successor).
            case_destinations: Successor ``caseDestinations`` (any successor).
            default_operands: Operand ``defaultOperands`` (any type). Empty by default.
            case_operands: Operand ``caseOperands`` (any type). Empty by default.
            case_values: Attribute ``case_values`` (integer elements attribute). Optional.
            branch_weights: Attribute ``branch_weights`` (i32 dense array attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: signless integer."""

    @property
    def default_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``defaultOperands``: any type."""

    @property
    def case_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``caseOperands``: any type."""

    @property
    def case_values(self) -> mlir_python._mlir_python.DenseIntElementsAttr | None:
        """Attribute ``case_values``: integer elements attribute."""

    @case_values.setter
    def case_values(
        self, arg: mlir_python._mlir_python.DenseIntElementsAttr | None
    ) -> None: ...
    @property
    def branch_weights(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``branch_weights``: i32 dense array attribute."""

    @branch_weights.setter
    def branch_weights(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def default_destination(self) -> mlir_python._mlir_python.Block:
        """Successor ``defaultDestination``."""

    @property
    def case_destinations(self) -> list[mlir_python._mlir_python.Block]:
        """Successor ``caseDestinations``."""

    OPERATION_NAME: str = "llvm.switch"

class TruncOp(mlir_python._mlir_python.Operation):
    """``llvm.trunc``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.trunc``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            overflow_flags: Property ``overflowFlags`` (flags of ``IntegerOverflowFlags``).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Property ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "llvm.trunc"

class UDivOp(mlir_python._mlir_python.Operation):
    """``llvm.udiv``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.udiv``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.udiv"

class UIToFPOp(mlir_python._mlir_python.Operation):
    """``llvm.uitofp``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        non_neg: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.uitofp``.

        Args:
            res_type: Type of result ``res`` (floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            non_neg: Attribute ``nonNeg`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type.
        """

    @property
    def non_neg(self) -> bool:
        """Attribute ``nonNeg``: unit attribute."""

    @non_neg.setter
    def non_neg(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.uitofp"

class URemOp(mlir_python._mlir_python.Operation):
    """``llvm.urem``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.urem``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.urem"

class UndefOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.undef``: Creates an undefined value of LLVM dialect type..

    Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
    must be created as SSA values using `llvm.mlir.undef`. This operation has no
    operands or attributes. It creates an undefined value of the specified LLVM
    IR dialect type.

    Example:

    ```mlir
    // Create a structure with a 32-bit integer followed by a float.
    %0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.undef``: Creates an undefined value of LLVM dialect type..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.mlir.undef"

class UnreachableOp(mlir_python._mlir_python.Operation):
    """``llvm.unreachable``."""

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """Create ``llvm.unreachable``."""

    OPERATION_NAME: str = "llvm.unreachable"

class VaArgOp(mlir_python._mlir_python.Operation):
    """``llvm.va_arg``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.va_arg``.

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            arg: Operand ``arg`` (LLVM pointer type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """Operand ``arg``: LLVM pointer type."""

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.va_arg"

class XOrOp(mlir_python._mlir_python.Operation):
    """``llvm.xor``."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.xor``.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            rhs: Operand ``rhs`` (signless integer or LLVM dialect-compatible vector of signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``rhs``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    OPERATION_NAME: str = "llvm.xor"

class ZExtOp(mlir_python._mlir_python.Operation):
    """``llvm.zext``."""

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        arg: mlir_python._mlir_python.Value,
        *,
        non_neg: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.zext``.

        Args:
            res_type: Type of result ``res`` (signless integer or LLVM dialect-compatible vector of signless integer).
            arg: Operand ``arg`` (signless integer or LLVM dialect-compatible vector of signless integer).
            non_neg: Attribute ``nonNeg`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``arg``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``res``: signless integer or LLVM dialect-compatible vector of signless integer.
        """

    @property
    def non_neg(self) -> bool:
        """Attribute ``nonNeg``: unit attribute."""

    @non_neg.setter
    def non_neg(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "llvm.zext"

class ZeroOp(mlir_python._mlir_python.Operation):
    """
    ``llvm.mlir.zero``: Creates a zero-initialized value of LLVM dialect type..

    Unlike LLVM IR, MLIR does not have first-class zero-initialized values.
    Such values must be created as SSA values using `llvm.mlir.zero`. This
    operation has no operands or attributes. It creates a zero-initialized
    value of the specified LLVM IR dialect type.

    Example:

    ```mlir
    // Create a zero-initialized value for a structure with a 32-bit integer
    // followed by a float.
    %0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
    ```
    """

    def __init__(
        self,
        res_type: mlir_python._mlir_python.Type,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``llvm.mlir.zero``: Creates a zero-initialized value of LLVM dialect type..

        Args:
            res_type: Type of result ``res`` (LLVM dialect-compatible type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def res(self) -> mlir_python._mlir_python.OpResult:
        """Result ``res``: LLVM dialect-compatible type."""

    OPERATION_NAME: str = "llvm.mlir.zero"

class PointerType(mlir_python._mlir_python.Type):
    """
    An opaque pointer, ``!llvm.ptr`` (or ``!llvm.ptr<N>`` in address
    space N). Pointers carry no pointee type; loads, stores, and
    address computations name the type they access.
    """

    def __init__(
        self,
        address_space: int = 0,
        *,
        context: mlir_python._mlir_python.Context | None = None,
    ) -> None:
        """Create a pointer type in ``address_space`` (0 is the default)."""

    @property
    def address_space(self) -> int:
        """The address space."""

class ArrayType(mlir_python._mlir_python.Type):
    """
    A fixed-size LLVM array, ``!llvm.array<N x T>``, e.g. the type of a
    string constant.
    """

    def __init__(self, element_type: mlir_python._mlir_python.Type, size: int) -> None:
        """Create ``!llvm.array<size x element_type>``."""

    @property
    def element_type(self) -> mlir_python._mlir_python.Type:
        """The element type."""

    @property
    def size(self) -> int:
        """The number of elements."""

class StructType(mlir_python._mlir_python.Type):
    """An LLVM structure, ``!llvm.struct<(T1, T2, ...)>``."""

    def __init__(
        self,
        elements: Sequence[mlir_python._mlir_python.Type],
        packed: bool = False,
        *,
        context: mlir_python._mlir_python.Context | None = None,
    ) -> None:
        """Create a literal struct of ``elements``; ``packed`` removes padding."""

    @property
    def elements(self) -> list[mlir_python._mlir_python.Type]:
        """The element types."""

    @property
    def packed(self) -> bool:
        """Whether the struct has no padding."""

class FunctionType(mlir_python._mlir_python.Type):
    """
    An LLVM function signature, ``!llvm.func<R (A, B, ...)>``, as taken
    by ``LLVMFuncOp``. Unlike the builtin ``FunctionType`` it has one
    result (``VoidType`` for none) and may be variadic, like ``printf``.
    """

    def __init__(
        self,
        result: mlir_python._mlir_python.Type,
        inputs: Sequence[mlir_python._mlir_python.Type],
        *,
        variadic: bool = False,
    ) -> None:
        """
        Create ``!llvm.func<result (inputs...)>``.

        Args:
            result: The result type; ``VoidType()`` for no result.
            inputs: The parameter types.
            variadic: Whether extra arguments may follow (``...``).
        """

    @property
    def result(self) -> mlir_python._mlir_python.Type:
        """The result type."""

    @property
    def inputs(self) -> list[mlir_python._mlir_python.Type]:
        """The parameter types."""

    @property
    def variadic(self) -> bool:
        """Whether the function takes extra arguments."""

class VoidType(mlir_python._mlir_python.Type):
    """``!llvm.void``: the result type of an LLVM function returning nothing."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!llvm.void``."""
