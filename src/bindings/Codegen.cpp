// Bindings for turning LLVM-dialect MLIR into LLVM IR, object code, and
// JIT-compiled functions.
#include "Core.h"

#include <cstring>
#include <fstream>

#include <llvm/ADT/bit.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/SwapByteOrder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

enum class OptLevel { O0 = 0, O1 = 1, O2 = 2, O3 = 3 };

llvm::CodeGenOptLevel codegenLevel(OptLevel level) {
  switch (level) {
  case OptLevel::O0:
    return llvm::CodeGenOptLevel::None;
  case OptLevel::O1:
    return llvm::CodeGenOptLevel::Less;
  case OptLevel::O2:
    return llvm::CodeGenOptLevel::Default;
  case OptLevel::O3:
    return llvm::CodeGenOptLevel::Aggressive;
  }
  return llvm::CodeGenOptLevel::Default;
}

void initializeNativeTarget() {
  static bool initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  }();
  (void)initialized;
}

/// A target machine for the host, emitting position-independent code.
std::unique_ptr<llvm::TargetMachine> hostTargetMachine(OptLevel level,
                                                       bool hostCpu) {
  initializeNativeTarget();
  llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    throw MLIRError("no LLVM target for " + triple.str() + ": " + error);
  std::string cpu = hostCpu ? llvm::sys::getHostCPUName().str() : "generic";
  std::string features;
  if (hostCpu)
    for (const auto &feature : llvm::sys::getHostCPUFeatures())
      features += (features.empty() ? "" : ",") +
                  std::string(feature.second ? "+" : "-") +
                  feature.first().str();
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple, cpu, features, llvm::TargetOptions(), llvm::Reloc::PIC_,
      std::nullopt, codegenLevel(level)));
  if (!machine)
    throw MLIRError("could not create an LLVM target machine for " +
                    triple.str());
  return machine;
}

std::string errorText(llvm::Error error) {
  return llvm::toString(std::move(error));
}

//===--------------------------------------------------------------------===//
// LLVM IR modules
//===--------------------------------------------------------------------===//

struct PyLLVMModule {
  // Declared first so it is destroyed after the module that uses it.
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
};

PyLLVMModule translate(const PyOperation &op) {
  mlir::Operation *native = get(op);
  DiagnosticCapture capture(native->getContext());
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      mlir::translateModuleToLLVMIR(native, *context, "mlir_python");
  if (!module)
    capture.raise("translation to LLVM IR failed; lower the module to the "
                  "LLVM dialect first (see codegen.lower_to_llvm)");
  std::unique_ptr<llvm::TargetMachine> machine =
      hostTargetMachine(OptLevel::O2, /*hostCpu=*/false);
  module->setDataLayout(machine->createDataLayout());
  module->setTargetTriple(machine->getTargetTriple());
  return PyLLVMModule{std::move(context), std::move(module)};
}

void emit(const PyLLVMModule &self, llvm::raw_pwrite_stream &out,
          llvm::CodeGenFileType kind, OptLevel level, bool hostCpu) {
  std::unique_ptr<llvm::TargetMachine> machine = hostTargetMachine(level, hostCpu);
  // Code generation mutates the module, so emit from a copy.
  std::unique_ptr<llvm::Module> copy = llvm::CloneModule(*self.module);
  copy->setDataLayout(machine->createDataLayout());
  llvm::legacy::PassManager passes;
  if (machine->addPassesToEmitFile(passes, out, nullptr, kind))
    throw MLIRError("the host target cannot emit this file type");
  passes.run(*copy);
}

//===--------------------------------------------------------------------===//
// JIT
//===--------------------------------------------------------------------===//

struct PyExecutionEngine {
  ContextHandle context;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

/// Storage for one argument or result passed through `invokePacked`.
using Slot = std::array<uint64_t, 2>;

std::string typeName(mlir::Type type) { return printToString(type); }

void checkScalar(mlir::Type type, const std::string &what) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
      integer && integer.getWidth() <= 64)
    return;
  if (llvm::isa<mlir::IndexType, mlir::Float32Type, mlir::Float64Type>(type))
    return;
  throw nb::type_error((what + " has type " + typeName(type) +
                        "; only integer (up to 64 bits), index, f32, and f64 "
                        "values, and memref arguments (as buffers), can be "
                        "passed between Python and compiled code")
                           .c_str());
}

void store(Slot &slot, mlir::Type type, nb::handle value, const std::string &what) {
  slot = {0, 0};
  if (llvm::isa<mlir::Float32Type>(type)) {
    float f = nb::cast<float>(value);
    std::memcpy(slot.data(), &f, sizeof f);
    return;
  }
  if (llvm::isa<mlir::Float64Type>(type)) {
    double d = nb::cast<double>(value);
    std::memcpy(slot.data(), &d, sizeof d);
    return;
  }
  unsigned width = llvm::isa<mlir::IndexType>(type)
                       ? 64
                       : llvm::cast<mlir::IntegerType>(type).getWidth();
  if (!nb::isinstance<nb::int_>(value))
    throw nb::type_error((what + " must be an int for " + typeName(type)).c_str());
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  bool isUnsigned = integer && integer.isUnsigned();
  bool isSigned = integer && integer.isSigned();
  // Signless integers accept both the signed and unsigned ranges.
  nb::int_ number = nb::borrow<nb::int_>(value);
  int64_t raw;
  if (width == 64) {
    if (nb::cast<bool>(number < nb::int_(0))) {
      if (isUnsigned)
        throw std::overflow_error(what + " must be non-negative");
      raw = nb::cast<int64_t>(number);
    } else if (isSigned) {
      raw = nb::cast<int64_t>(number);
    } else {
      raw = static_cast<int64_t>(nb::cast<uint64_t>(number));
    }
  } else {
    int64_t v = nb::cast<int64_t>(number);
    int64_t low = isUnsigned ? 0 : -(int64_t(1) << (width - 1));
    int64_t high = isSigned ? (int64_t(1) << (width - 1)) - 1
                            : (int64_t(1) << width) - 1;
    if (width == 1) {
      low = isSigned ? -1 : 0;
      high = 1;
    }
    if (v < low || v > high)
      throw std::overflow_error(what + " = " + std::to_string(v) +
                                " does not fit " + typeName(type));
    raw = v;
  }
  std::memcpy(slot.data(), &raw, (width + 7) / 8);
}

/// A Python buffer (NumPy array, array.array, memoryview, ...) held for the
/// duration of a call and checked against the memref type it is passed as.
class HeldBuffer {
public:
  HeldBuffer(nb::handle object, mlir::MemRefType type, const std::string &what) {
    if (PyObject_GetBuffer(object.ptr(), &view_, PyBUF_RECORDS) != 0) {
      PyErr_Clear();
      throw nb::type_error((what + " must be a writable buffer (e.g. a NumPy array) "
                                   "for " +
                            typeName(type))
                               .c_str());
    }
    held_ = true;
    check(type, what);
  }
  HeldBuffer(const HeldBuffer &) = delete;
  HeldBuffer &operator=(const HeldBuffer &) = delete;
  ~HeldBuffer() {
    if (held_)
      PyBuffer_Release(&view_);
  }

  [[nodiscard]] void *data() const { return view_.buf; }
  [[nodiscard]] int64_t size(int dim) const { return view_.shape[dim]; }
  [[nodiscard]] int64_t stride(int dim) const {
    return view_.strides[dim] / view_.itemsize;
  }

private:
  void check(mlir::MemRefType type, const std::string &what) {
    auto fail = [&](const std::string &problem) {
      throw nb::type_error((what + " (" + typeName(type) + "): " + problem).c_str());
    };
    if (type.getMemorySpace())
      fail("only memrefs in the default memory space can be passed");
    if (view_.ndim != type.getRank())
      fail("expected " + std::to_string(type.getRank()) + " dimensions, got " +
           std::to_string(view_.ndim));
    for (int64_t i = 0; i < type.getRank(); ++i)
      if (!type.isDynamicDim(i) && type.getDimSize(i) != view_.shape[i])
        fail("dimension " + std::to_string(i) + " must be " +
             std::to_string(type.getDimSize(i)) + ", got " +
             std::to_string(view_.shape[i]));
    checkElement(type.getElementType(), fail);
    for (int64_t i = 0; i < type.getRank(); ++i)
      if (view_.strides[i] % view_.itemsize != 0)
        fail("strides must be multiples of the element size");
    // Static strides are compiled in as constants, so the buffer must match.
    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    if (mlir::failed(type.getStridesAndOffset(strides, offset)))
      fail("layouts without strides cannot be passed");
    for (int64_t i = 0; i < type.getRank(); ++i)
      if (!mlir::ShapedType::isDynamic(strides[i]) && strides[i] != stride(i))
        fail("stride of dimension " + std::to_string(i) + " must be " +
             std::to_string(strides[i]) + " elements, got " +
             std::to_string(stride(i)) + " (pass a contiguous copy)");
    if (!mlir::ShapedType::isDynamic(offset) && offset != 0)
      fail("layouts with a nonzero static offset cannot be passed");
  }

  template <typename Fail> void checkElement(mlir::Type element, Fail &fail) {
    std::string format = view_.format ? view_.format : "B";
    if (!format.empty() && (format[0] == '@' || format[0] == '=' ||
                            (format[0] == '<' && llvm::sys::IsLittleEndianHost)))
      format.erase(0, 1);
    auto size = static_cast<size_t>(view_.itemsize);
    bool isFloat = format == "f" || format == "d" || format == "e";
    bool isSignedInt = format.size() == 1 && std::strchr("bhilqn", format[0]);
    bool isUnsignedInt = format.size() == 1 && std::strchr("BHILQN", format[0]);
    std::string got = "'" + format + "' items of " + std::to_string(size) + " bytes";
    if (llvm::isa<mlir::Float32Type>(element)) {
      if (!(format == "f" && size == 4))
        fail("expected float32 items, got " + got);
      return;
    }
    if (llvm::isa<mlir::Float64Type>(element)) {
      if (!(format == "d" && size == 8))
        fail("expected float64 items, got " + got);
      return;
    }
    (void)isFloat;
    auto integer = llvm::dyn_cast<mlir::IntegerType>(element);
    if (integer && integer.getWidth() == 1) {
      if (!(format == "?" && size == 1))
        fail("expected bool items, got " + got);
      return;
    }
    unsigned width = llvm::isa<mlir::IndexType>(element) ? 64
                     : integer                          ? integer.getWidth()
                                                        : 0;
    if (width == 0 || width % 8 != 0 || width > 64)
      fail("element type " + typeName(element) + " cannot be passed from Python");
    bool signOk = integer && integer.isSigned()     ? isSignedInt
                  : integer && integer.isUnsigned() ? isUnsignedInt
                                                    : isSignedInt || isUnsignedInt;
    if (!signOk || size * 8 != width)
      fail("expected " + std::to_string(width) + "-bit " +
           (integer && integer.isUnsigned() ? "unsigned "
            : integer && integer.isSigned() ? "signed "
                                            : "") +
           "integer items, got " + got);
  }

  Py_buffer view_{};
  bool held_ = false;
};

/// Size in bytes of a scalar as LLVM stores it (also its natural alignment).
size_t storageSize(mlir::Type type) {
  if (llvm::isa<mlir::Float32Type>(type))
    return 4;
  if (llvm::isa<mlir::Float64Type, mlir::IndexType>(type))
    return 8;
  return llvm::PowerOf2Ceil((llvm::cast<mlir::IntegerType>(type).getWidth() + 7) / 8);
}

nb::object load(const char *bytes, mlir::Type type) {
  if (llvm::isa<mlir::Float32Type>(type)) {
    float f;
    std::memcpy(&f, bytes, sizeof f);
    return nb::float_(f);
  }
  if (llvm::isa<mlir::Float64Type>(type)) {
    double d;
    std::memcpy(&d, bytes, sizeof d);
    return nb::float_(d);
  }
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  unsigned width = integer ? integer.getWidth() : 64;
  uint64_t raw = 0;
  std::memcpy(&raw, bytes, (width + 7) / 8);
  if (width < 64)
    raw &= (uint64_t(1) << width) - 1;
  if (width == 1)
    return nb::bool_(raw != 0);
  if (integer && integer.isUnsigned())
    return nb::int_(raw);
  // Signless and signed integers are read as signed.
  int64_t value = static_cast<int64_t>(raw << (64 - width)) >> (64 - width);
  return nb::int_(value);
}

nb::object call(PyExecutionEngine &self, const std::string &name,
                const PyFunctionType &signature, const std::vector<nb::object> &args) {
  auto type = llvm::cast<mlir::FunctionType>(signature.type);
  if (args.size() != type.getNumInputs())
    throw nb::type_error((name + "() takes " + std::to_string(type.getNumInputs()) +
                          " arguments, got " + std::to_string(args.size()))
                             .c_str());
  // A memref argument is passed as its lowered descriptor: allocated and
  // aligned pointers, offset, then one size and one stride per dimension.
  size_t slotCount = 0;
  for (mlir::Type input : type.getInputs())
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(input))
      slotCount += 3 + 2 * memref.getRank();
    else
      slotCount += 1;
  std::vector<Slot> slots(slotCount, Slot{0, 0});
  std::vector<std::unique_ptr<HeldBuffer>> buffers;
  std::vector<void *> pointers;
  auto next = [&]() -> Slot & {
    Slot &slot = slots[pointers.size()];
    pointers.push_back(slot.data());
    return slot;
  };
  auto storeInt = [](Slot &slot, int64_t value) {
    std::memcpy(slot.data(), &value, sizeof value);
  };
  for (unsigned i = 0; i < type.getNumInputs(); ++i) {
    std::string what = "argument " + std::to_string(i);
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type.getInput(i))) {
      buffers.push_back(std::make_unique<HeldBuffer>(args[i], memref, what));
      const HeldBuffer &buffer = *buffers.back();
      void *data = buffer.data();
      std::memcpy(next().data(), &data, sizeof data); // allocated
      std::memcpy(next().data(), &data, sizeof data); // aligned
      storeInt(next(), 0);                            // offset
      for (int64_t d = 0; d < memref.getRank(); ++d)
        storeInt(next(), buffer.size(static_cast<int>(d)));
      for (int64_t d = 0; d < memref.getRank(); ++d)
        storeInt(next(), buffer.stride(static_cast<int>(d)));
      continue;
    }
    checkScalar(type.getInput(i), what);
    store(next(), type.getInput(i), args[i], what);
  }
  // Results are returned through one pointer: to the value itself, or for
  // several results to an LLVM struct of them (naturally aligned fields).
  std::vector<size_t> offsets;
  size_t resultBytes = 0;
  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    checkScalar(type.getResult(i), "result " + std::to_string(i));
    size_t size = storageSize(type.getResult(i));
    resultBytes = llvm::alignTo(resultBytes, size);
    offsets.push_back(resultBytes);
    resultBytes += size;
  }
  std::vector<uint64_t> resultStorage((resultBytes + 7) / 8 + 1, 0);
  if (type.getNumResults() > 0)
    pointers.push_back(resultStorage.data());
  llvm::Error error = [&] {
    nb::gil_scoped_release release;
    return self.engine->invokePacked(name, pointers);
  }();
  if (error)
    throw MLIRError("calling '" + name + "' failed: " + errorText(std::move(error)));
  const char *bytes = reinterpret_cast<const char *>(resultStorage.data());
  if (type.getNumResults() == 0)
    return nb::none();
  if (type.getNumResults() == 1)
    return load(bytes, type.getResult(0));
  nb::list results;
  for (unsigned i = 0; i < type.getNumResults(); ++i)
    results.append(load(bytes + offsets[i], type.getResult(i)));
  return nb::tuple(results);
}

} // namespace

void bindCodegen(nb::module_ &m) {
  nb::enum_<OptLevel>(m, "OptLevel", "LLVM optimization level.")
      .value("O0", OptLevel::O0, "No optimization.")
      .value("O1", OptLevel::O1, "Light optimization.")
      .value("O2", OptLevel::O2, "Default optimization.")
      .value("O3", OptLevel::O3, "Aggressive optimization.");

  nb::class_<PyLLVMModule>(
      m, "LLVMModule",
      "An LLVM IR module translated from LLVM-dialect MLIR, targeting the\n"
      "host (its triple and data layout are set).")
      .def("__str__",
           [](const PyLLVMModule &self) {
             std::string text;
             llvm::raw_string_ostream stream(text);
             self.module->print(stream, nullptr);
             return text;
           })
      .def(
          "verify",
          [](const PyLLVMModule &self) {
            std::string problems;
            llvm::raw_string_ostream stream(problems);
            if (llvm::verifyModule(*self.module, &stream))
              throw MLIRError("LLVM IR verification failed:\n" + problems);
          },
          "Run LLVM's IR verifier.\n\nRaises:\n"
          "    MLIRError: With the verifier's findings.")
      .def_prop_ro(
          "defined_functions",
          [](const PyLLVMModule &self) {
            std::vector<std::string> names;
            for (const llvm::Function &function : *self.module)
              if (!function.isDeclaration())
                names.push_back(function.getName().str());
            return names;
          },
          "Names of the functions with a body, in module order.")
      .def_prop_ro(
          "target_triple",
          [](const PyLLVMModule &self) {
            return self.module->getTargetTriple().str();
          },
          "The target triple, e.g. ``x86_64-unknown-linux-gnu``.")
      .def_prop_ro(
          "data_layout",
          [](const PyLLVMModule &self) {
            return self.module->getDataLayoutStr();
          },
          "The data layout string.")
      .def(
          "optimize",
          [](PyLLVMModule &self, OptLevel level) {
            std::unique_ptr<llvm::TargetMachine> machine =
                hostTargetMachine(level, /*hostCpu=*/false);
            auto transformer = mlir::makeOptimizingTransformer(
                static_cast<unsigned>(level), 0, machine.get());
            if (llvm::Error error = transformer(self.module.get()))
              throw MLIRError("LLVM optimization failed: " +
                              errorText(std::move(error)));
          },
          "level"_a = OptLevel::O2, "Run LLVM's optimization pipeline in place.")
      .def(
          "write_object",
          [](const PyLLVMModule &self, const std::string &path, OptLevel level,
             bool hostCpu) {
            std::error_code error;
            llvm::raw_fd_ostream out(path, error);
            if (error)
              throw MLIRError(path + ": " + error.message());
            emit(self, out, llvm::CodeGenFileType::ObjectFile, level, hostCpu);
          },
          "path"_a, nb::kw_only(), "level"_a = OptLevel::O2,
          "host_cpu"_a = false,
          "Write a position-independent object file for the host, linkable\n"
          "into executables and shared libraries.\n\n"
          "Args:\n"
          "    path: Output file.\n"
          "    level: Code generation optimization level.\n"
          "    host_cpu: Use every feature of this machine's CPU; the object\n"
          "        may then not run on other machines.")
      .def(
          "assembly",
          [](const PyLLVMModule &self, OptLevel level, bool hostCpu) {
            llvm::SmallVector<char, 0> buffer;
            llvm::raw_svector_ostream out(buffer);
            emit(self, out, llvm::CodeGenFileType::AssemblyFile, level, hostCpu);
            return std::string(buffer.begin(), buffer.end());
          },
          nb::kw_only(), "level"_a = OptLevel::O2, "host_cpu"_a = false,
          "The host assembly for this module.");

  m.def("translate_to_llvm_ir", &translate, "module"_a,
        "Translate an MLIR module in the LLVM dialect to LLVM IR.\n\n"
        "Raises:\n"
        "    MLIRError: If the module still contains non-LLVM operations.");

  nb::class_<PyExecutionEngine>(
      m, "ExecutionEngine",
      "JIT-compiles an MLIR module in the LLVM dialect into this process.\n\n"
      "Prefer ``codegen.compile``, which lowers a copy of a module and calls\n"
      "functions by their ``func.FuncOp``.")
      .def(nb::new_([](const PyOperation &module, OptLevel level,
                       const std::vector<std::string> &sharedLibraries) {
             initializeNativeTarget();
             mlir::Operation *native = get(module);
             std::function<llvm::Error(llvm::Module *)> transformer =
                 mlir::makeOptimizingTransformer(static_cast<unsigned>(level),
                                                 0, nullptr);
             llvm::SmallVector<llvm::StringRef> libraries(sharedLibraries.begin(),
                                                          sharedLibraries.end());
             mlir::ExecutionEngineOptions options;
             options.transformer = transformer;
             options.jitCodeGenOptLevel = codegenLevel(level);
             options.sharedLibPaths = libraries;
             DiagnosticCapture capture(native->getContext());
             auto created = mlir::ExecutionEngine::create(native, options);
             if (!created) {
               std::string reason = errorText(created.takeError());
               capture.raise("JIT compilation failed: " + reason);
             }
             (*created)->initialize();
             return PyExecutionEngine{module.tree->context(), std::move(*created)};
           }),
           "module"_a, nb::kw_only(), "opt_level"_a = OptLevel::O2,
           "shared_libraries"_a = std::vector<std::string>{},
           "JIT-compile ``module``.\n\n"
           "Args:\n"
           "    module: A module in the LLVM dialect.\n"
           "    opt_level: LLVM optimization level.\n"
           "    shared_libraries: Libraries to load for external symbols.\n\n"
           "Raises:\n    MLIRError: If translation or compilation fails.")
      .def("call", &call, "name"_a, "signature"_a, "args"_a,
           nb::sig("def call(self, name: str, signature: FunctionType, "
                   "args: collections.abc.Sequence[int | float | bool | "
                   "collections.abc.Buffer], /) -> "
                   "int | float | bool | tuple[int | float | bool, ...] | None"),
           "Low level: call function ``name`` of type ``signature`` with\n"
           "``args``: Python numbers for scalars, writable buffers (such as\n"
           "NumPy arrays) for memrefs, which the function reads and writes in\n"
           "place. Returns ``None``, one value, or a tuple of values.\n\n"
           "Raises:\n"
           "    TypeError: For a wrong argument count, argument types, or\n"
           "        types that cannot cross the Python boundary.\n"
           "    OverflowError: If an int does not fit its integer type.\n"
           "    MLIRError: If no such function exists.");
}

} // namespace mlir_python
