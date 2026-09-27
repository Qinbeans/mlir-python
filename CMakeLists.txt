cmake_minimum_required(VERSION 3.24)

project(mlir_python LANGUAGES CXX)

if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(Python_EXECUTABLE
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/python3/python3"
        CACHE FILEPATH "vcpkg Python interpreter" FORCE
    )
endif()
find_package(Python 3.12 REQUIRED COMPONENTS
    Interpreter
    Development.Module
    Development.SABIModule
)
find_package(nanobind CONFIG REQUIRED)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
# Code generation targets the machine building the wheel.
set(LLVM_TARGETS_TO_BUILD "Native" CACHE STRING "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_BUILD_UTILS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "")
add_subdirectory(third_party/llvm-project/llvm EXCLUDE_FROM_ALL)

set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir)

# Generated Python is formatted with the project's ruff configuration when ruff
# is available (it is a dev dependency), so regeneration matches the linter.
find_program(MLIR_PYTHON_RUFF ruff HINTS ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin)
set(MLIR_PYTHON_RUFF_ARGS "")
if(MLIR_PYTHON_RUFF)
    set(MLIR_PYTHON_RUFF_ARGS --ruff ${MLIR_PYTHON_RUFF})
else()
    message(STATUS "ruff not found; generated Python will not be formatted")
endif()
set(MLIR_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir_python)
set(MLIR_PYTHON_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(MLIR_PYTHON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${MLIR_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
)

#-------------------------------------------------------------------------------
# Linked dialects
#
# Each entry is: name, ODS file (under mlir/include/mlir/Dialect), C++ header,
# C++ namespace. Typed Python bindings are generated for every entry. Keep the
# libraries below and src/bindings/Registration.cpp in sync.
#-------------------------------------------------------------------------------
set(MLIR_PYTHON_DIALECTS
    "arith|Arith/IR/ArithOps.td|mlir/Dialect/Arith/IR/Arith.h|::mlir::arith"
    "async|Async/IR/AsyncOps.td|mlir/Dialect/Async/IR/Async.h|::mlir::async"
    "bufferization|Bufferization/IR/BufferizationOps.td|mlir/Dialect/Bufferization/IR/Bufferization.h|::mlir::bufferization"
    "cf|ControlFlow/IR/ControlFlowOps.td|mlir/Dialect/ControlFlow/IR/ControlFlowOps.h|::mlir::cf"
    "emitc|EmitC/IR/EmitC.td|mlir/Dialect/EmitC/IR/EmitC.h|::mlir::emitc"
    "func|Func/IR/FuncOps.td|mlir/Dialect/Func/IR/FuncOps.h|::mlir::func"
    "irdl|IRDL/IR/IRDLOps.td|mlir/Dialect/IRDL/IR/IRDL.h|::mlir::irdl"
    "llvm|LLVMIR/LLVMOps.td|mlir/Dialect/LLVMIR/LLVMDialect.h|::mlir::LLVM"
    "math|Math/IR/MathOps.td|mlir/Dialect/Math/IR/Math.h|::mlir::math"
    "memref|MemRef/IR/MemRefOps.td|mlir/Dialect/MemRef/IR/MemRef.h|::mlir::memref"
    "scf|SCF/IR/SCFOps.td|mlir/Dialect/SCF/IR/SCF.h|::mlir::scf"
    "tensor|Tensor/IR/TensorOps.td|mlir/Dialect/Tensor/IR/Tensor.h|::mlir::tensor"
    "vector|Vector/IR/VectorOps.td|mlir/Dialect/Vector/IR/VectorOps.h|::mlir::vector"
)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    set(MLIR_PYTHON_NATIVE_ARCH X86)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(MLIR_PYTHON_NATIVE_ARCH AArch64)
else()
    message(FATAL_ERROR "unsupported host processor ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
set(MLIR_PYTHON_MLIR_LIBRARIES
    LLVM${MLIR_PYTHON_NATIVE_ARCH}AsmParser
    LLVM${MLIR_PYTHON_NATIVE_ARCH}CodeGen
    LLVM${MLIR_PYTHON_NATIVE_ARCH}Desc
    LLVM${MLIR_PYTHON_NATIVE_ARCH}Info
    LLVMPasses
    MLIRArithDialect
    MLIRArithTransforms
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRAsyncTransforms
    MLIRBufferizationDialect
    MLIRBufferizationPipelines
    MLIRBufferizationToMemRef
    MLIRBufferizationTransforms
    MLIRArithToEmitC
    MLIRArithToLLVM
    MLIRBuiltinToLLVMIRTranslation
    MLIRComplexToLLVM
    MLIRControlFlowToLLVM
    MLIRControlFlowToSCF
    MLIRConvertToEmitC
    MLIRConvertToLLVMPass
    MLIREmitCDialect
    MLIREmitCTransforms
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
    MLIRFuncToEmitC
    MLIRFuncToLLVM
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMathToEmitC
    MLIRMathToLLVM
    MLIRMemRefToEmitC
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFToEmitC
    MLIRTargetCpp
    MLIRTargetLLVMIRExport
    MLIRUBToLLVM
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRVectorToSCF
    MLIRControlFlowDialect
    MLIRControlFlowTransforms
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRInferTypeOpInterface
    MLIRIRDL
    MLIRIR
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
)

# LLVM is built without RTTI, so classes deriving from its classes (passes)
# must be too; the rest of the extension keeps RTTI for nanobind.
if(NOT LLVM_ENABLE_RTTI AND NOT MSVC)
    set_source_files_properties(src/bindings/ProjectPasses.cpp
        PROPERTIES COMPILE_OPTIONS -fno-rtti)
endif()

# Reports facts about the compiled operations that ODS records leave implicit.
add_executable(mlir-python-introspect
    tools/introspect.cpp
    src/bindings/ProjectPasses.cpp
    src/bindings/Registration.cpp
)
target_compile_features(mlir-python-introspect PRIVATE cxx_std_17)
target_include_directories(mlir-python-introspect PRIVATE ${MLIR_PYTHON_INCLUDE_DIRS})
target_link_libraries(mlir-python-introspect PRIVATE ${MLIR_PYTHON_MLIR_LIBRARIES})

set(MLIR_PYTHON_INTROSPECTION ${MLIR_PYTHON_GENERATED_DIR}/introspection.json)

set(namespaces "")
foreach(entry IN LISTS MLIR_PYTHON_DIALECTS)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 name)
    list(GET fields 3 namespace)
    list(APPEND namespaces "${name}=${namespace}")
endforeach()
list(JOIN namespaces "," MLIR_PYTHON_DIALECT_NAMESPACES)

set(MLIR_PYTHON_GENERATED_SOURCES "")
set(MLIR_PYTHON_CLASS_MAPS "")
set(MLIR_PYTHON_GENERATED_MODULES "")
set(dialect_declarations "")
set(dialect_calls "")
foreach(entry IN LISTS MLIR_PYTHON_DIALECTS)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 name)
    list(GET fields 1 td)
    list(GET fields 2 header)
    set(td_path ${MLIR_SOURCE_DIR}/include/mlir/Dialect/${td})
    set(records ${MLIR_PYTHON_GENERATED_DIR}/${name}.json)
    set(cpp ${MLIR_PYTHON_GENERATED_DIR}/${name}.cpp)
    # Dialects named like Python keywords get modules named
    # <name>_dialect (mlir_python.dialects.async_dialect), as
    # tools/gen_dialects.py does.
    set(pyname ${name})
    if(name STREQUAL "async")
        set(pyname async_dialect)
    endif()
    set(py ${MLIR_PYTHON_PACKAGE_DIR}/dialects/${pyname}.py)
    set(classes ${MLIR_PYTHON_GENERATED_DIR}/${name}.classes.json)
    add_custom_command(
        OUTPUT ${records}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_PYTHON_GENERATED_DIR}
        COMMAND llvm-tblgen --dump-json -I ${MLIR_SOURCE_DIR}/include ${td_path}
            -o ${records}
        DEPENDS llvm-tblgen ${td_path}
        COMMENT "Dumping ODS records for '${name}'"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${cpp} ${py} ${classes}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_dialects.py
            --name ${name} --records ${records}
            --introspection ${MLIR_PYTHON_INTROSPECTION}
            --include ${header} --namespaces ${MLIR_PYTHON_DIALECT_NAMESPACES}
            --output-cpp ${cpp} --output-py ${py} --output-classes ${classes}
            ${MLIR_PYTHON_RUFF_ARGS}
        DEPENDS ${records} ${MLIR_PYTHON_INTROSPECTION}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_dialects.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/pyformat.py
        COMMENT "Generating typed bindings for '${name}'"
        VERBATIM
    )
    list(APPEND MLIR_PYTHON_GENERATED_SOURCES ${cpp})
    list(APPEND MLIR_PYTHON_CLASS_MAPS ${classes})
    list(APPEND MLIR_PYTHON_GENERATED_MODULES ${pyname})
    string(APPEND dialect_declarations "void bindDialect_${name}(nb::module_ &parent);\n")
    string(APPEND dialect_calls "  bindDialect_${name}(m);\n")
endforeach()

#-------------------------------------------------------------------------------
# Linked passes
#
# Pass definition files (under mlir/include/mlir) whose passes are registered
# by src/bindings/Registration.cpp. mlir_python.passes gets one typed class per
# pass defined in them.
#-------------------------------------------------------------------------------
set(MLIR_PYTHON_PASS_DEFINITIONS
    Conversion/Passes.td
    Dialect/Async/Passes.td
    Dialect/Bufferization/Transforms/Passes.td
    Dialect/EmitC/Transforms/Passes.td
    Dialect/MemRef/Transforms/Passes.td
    Dialect/Vector/Transforms/Passes.td
    Transforms/Passes.td
)
set(pass_records "")
foreach(td IN LISTS MLIR_PYTHON_PASS_DEFINITIONS)
    string(MAKE_C_IDENTIFIER "${td}" id)
    set(td_path ${MLIR_SOURCE_DIR}/include/mlir/${td})
    set(records ${MLIR_PYTHON_GENERATED_DIR}/passes_${id}.json)
    add_custom_command(
        OUTPUT ${records}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_PYTHON_GENERATED_DIR}
        COMMAND llvm-tblgen --dump-json -I ${MLIR_SOURCE_DIR}/include ${td_path}
            -o ${records}
        DEPENDS llvm-tblgen ${td_path}
        COMMENT "Dumping pass records for '${td}'"
        VERBATIM
    )
    list(APPEND pass_records ${records})
endforeach()
# Passes mlir-python defines (implemented in src/bindings/ProjectPasses.cpp).
set(project_pass_td ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/ProjectPasses.td)
set(records ${MLIR_PYTHON_GENERATED_DIR}/passes_project.json)
add_custom_command(
    OUTPUT ${records}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_PYTHON_GENERATED_DIR}
    COMMAND llvm-tblgen --dump-json -I ${MLIR_SOURCE_DIR}/include ${project_pass_td}
        -o ${records}
    DEPENDS llvm-tblgen ${project_pass_td}
    COMMENT "Dumping pass records for mlir-python's own passes"
    VERBATIM
)
list(APPEND pass_records ${records})
add_custom_command(
    OUTPUT ${MLIR_PYTHON_INTROSPECTION}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_PYTHON_GENERATED_DIR}
    COMMAND mlir-python-introspect ${MLIR_PYTHON_INTROSPECTION} ${pass_records}
    DEPENDS mlir-python-introspect ${pass_records}
    COMMENT "Introspecting linked MLIR dialects"
    VERBATIM
)
set(MLIR_PYTHON_PASSES_MODULE ${MLIR_PYTHON_PACKAGE_DIR}/passes.py)
add_custom_command(
    OUTPUT ${MLIR_PYTHON_PASSES_MODULE}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_passes.py
        --records ${pass_records} --classes ${MLIR_PYTHON_CLASS_MAPS}
        --introspection ${MLIR_PYTHON_INTROSPECTION}
        --output ${MLIR_PYTHON_PASSES_MODULE} ${MLIR_PYTHON_RUFF_ARGS}
    DEPENDS ${pass_records} ${MLIR_PYTHON_CLASS_MAPS} ${MLIR_PYTHON_INTROSPECTION}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_passes.py
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/pyformat.py
    COMMENT "Generating typed passes"
    VERBATIM
)
add_custom_target(mlir_python_passes ALL DEPENDS ${MLIR_PYTHON_PASSES_MODULE})

file(CONFIGURE OUTPUT ${MLIR_PYTHON_GENERATED_DIR}/BindDialects.cpp CONTENT
"// Generated by CMakeLists.txt. Do not edit.
#include \"Core.h\"

namespace mlir_python {

@dialect_declarations@
void bindDialects(nb::module_ &m) {
@dialect_calls@}

} // namespace mlir_python
" @ONLY)

#-------------------------------------------------------------------------------
# Extension module
#-------------------------------------------------------------------------------
nanobind_add_module(_mlir_python STABLE_ABI
    src/bindings/Attributes.cpp
    src/bindings/Codegen.cpp
    src/bindings/Context.cpp
    src/bindings/Core.cpp
    src/bindings/DialectExtras.cpp
    src/bindings/IR.cpp
    src/bindings/Module.cpp
    src/bindings/Passes.cpp
    src/bindings/ProjectPasses.cpp
    src/bindings/Registration.cpp
    src/bindings/Types.cpp
    ${MLIR_PYTHON_GENERATED_DIR}/BindDialects.cpp
    ${MLIR_PYTHON_GENERATED_SOURCES}
)
target_compile_features(_mlir_python PRIVATE cxx_std_17)
target_include_directories(_mlir_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings
    ${MLIR_PYTHON_INCLUDE_DIRS}
)
target_link_libraries(_mlir_python PRIVATE ${MLIR_PYTHON_MLIR_LIBRARIES})

install(TARGETS _mlir_python LIBRARY DESTINATION mlir_python)

# MLIR's reference runtime for the async dialect (thread pool, tokens,
# values, groups), shipped next to the extension. Compiled async code calls
# it through a C API, so another implementation can replace it (see
# mlir_python.codegen.async_runtime).
# Unversioned (libmlir_async_runtime.so), since wheels cannot hold the
# versioned name's symlinks.
set_property(TARGET mlir_async_runtime PROPERTY VERSION)
set_property(TARGET mlir_async_runtime PROPERTY SOVERSION)
add_dependencies(_mlir_python mlir_async_runtime)
install(TARGETS mlir_async_runtime LIBRARY DESTINATION mlir_python)

# Development builds place the module next to the Python sources, so
# `import mlir_python` works from src/, and regenerate its type stubs.
option(MLIR_PYTHON_INPLACE
    "Copy the built module into src/mlir_python and regenerate its stubs" OFF)
if(MLIR_PYTHON_INPLACE)
    add_custom_command(TARGET _mlir_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:_mlir_python> ${MLIR_PYTHON_PACKAGE_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:mlir_async_runtime> ${MLIR_PYTHON_PACKAGE_DIR}/
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/register_venv.cmake
        VERBATIM
    )
    set(stub_dir ${MLIR_PYTHON_PACKAGE_DIR}/_mlir_python)
    set(stubs ${stub_dir}/__init__.pyi)
    foreach(name IN LISTS MLIR_PYTHON_GENERATED_MODULES)
        list(APPEND stubs ${stub_dir}/${name}.pyi)
    endforeach()
    nanobind_add_stub(_mlir_python_stub
        RECURSIVE
        MODULE mlir_python._mlir_python
        OUTPUT_PATH ${MLIR_PYTHON_PACKAGE_DIR}
        OUTPUT ${stubs}
        PYTHON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDS _mlir_python
    )
    if(MLIR_PYTHON_RUFF)
        add_custom_command(TARGET _mlir_python_stub POST_BUILD
            COMMAND ${MLIR_PYTHON_RUFF} check --fix --exit-zero --quiet ${stub_dir}
            COMMAND ${MLIR_PYTHON_RUFF} format --quiet ${stub_dir}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM
        )
    endif()
endif()
