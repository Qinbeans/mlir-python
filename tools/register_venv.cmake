# Makes src/ importable from the project's virtual environment (.venv), so
# `uv run script.py` and plain `python` find the in-place build like pytest
# does. Run by the MLIR_PYTHON_INPLACE post-build step:
#   cmake -DSOURCE_DIR=<repo> -P tools/register_venv.cmake
file(GLOB site_dirs
    "${SOURCE_DIR}/.venv/lib/python3*/site-packages"
    "${SOURCE_DIR}/.venv/Lib/site-packages"
)
foreach(dir IN LISTS site_dirs)
    # Written only when the content changes.
    file(CONFIGURE OUTPUT "${dir}/mlir_python_dev.pth" CONTENT "${SOURCE_DIR}/src\n")
endforeach()
