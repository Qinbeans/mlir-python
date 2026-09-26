from mlir_python.lang import Program, cstr, i32, stack

program = Program()


@program.extern
def puts(s: cstr) -> i32: ...


@program.extern
def printf(format: cstr, *args) -> i32: ...


@program.extern
def scanf(format: cstr, *args) -> i32: ...


@program.main
def main() -> None:
    value = stack(i32)
    puts("Enter a value:")
    scanf("%d\n", value)
    printf("Hello, world! The answer is %d\n", value[0])


program.build_executable("hello")
print(program)
