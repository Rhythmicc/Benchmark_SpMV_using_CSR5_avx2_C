# Benchmark_SpMV_using_CSR5 (C)

Refer from: https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5

Benchmark_SpMV_using_CSR5_avx2 with `C`

## Build

```shell
cd Benchmark_SpMV_using_CSR5_avx2_C
qrun -b # with Qpro
# or
cd dist
cmake ..
make -j
```

## Run

```shell
./dist/C_CSR5 Alemdar.mtx
```

