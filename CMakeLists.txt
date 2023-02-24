cmake_minimum_required(VERSION 3.17)
project(C_CSR5 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER gcc)

add_compile_options(-mavx -mavx2 -march=native -O3)
add_compile_options(-Xpreprocessor ${OpenMP_C_FLAGS} -mfma)

include_directories("CSR5_avx2")
link_libraries(m gomp)
add_executable(C_CSR5 main.c CSR5_avx2 anonymouslib_avx2.h mmio.h)
