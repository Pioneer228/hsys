# hsys

## Build

### Ninja
```bash
cmake -B build/ -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/
```

### Make
```bash
cmake -B build/ -DCMAKE_BUILD_TYPE=Release
cmake --build build/ -j8
```

## Tests

```bash
ctest --progress --test-dir build/work1/tests/ -R 'VectorTest|VectorAddTest'
```

## Benchmarks
```bash
./build/work1/benchmarks/work1_benchmarks --benchmark_format=json > benchmarks_results_work1.json

python ./work1/utils/benchmark_pp/complexity_chart.py \
    -j benchmarks_results_work1.json \
    --xlog --ylog -c complexity_chart.ht

python ./work1/utils/benchmark_pp/speedup_chart.py \
    -j benchmarks_results_work1.json \
    -r 'Eigen Vector Addition (CPU)' \
    -t 'CUDA Vector Addition (GPU)' \
    --xlog --ylog -c speedup_chart.html
```
___
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lF-1fwBhbl3j1S9p1BGRQgu7AMCJS6UY?usp=sharing)
