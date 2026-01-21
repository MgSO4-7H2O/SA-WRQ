# SA-WRQ ANN Skeleton

This repository contains a contract-first skeleton for an online RVQ + IVF retrieval system with optional whitening, dual-route recall, and monitoring. All modules expose strict C++17 interfaces that are ready for concurrent development and testing.

## Build & Test

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

## Running the CLIs

```bash
# Build index workflow
./build_index ../configs/base.json [base_fvecs_or_directory]

# Evaluation workflow
./run_eval ../configs/base.json [dataset_dir_or_base_fvecs] [query_fvecs]
```

- When `.fvecs` paths (FAISS format) are provided, the tools load them through `LoadFvecs` (row-major) instead of generating toy random data.
- For convenience, you can pass a dataset directory (e.g., `data/cifar60k`) and the tools will look for files ending with `_base.fvecs` and `_query.fvecs`.
- Both binaries always accept a path to a JSON config file matching the fields defined in `include/common/config.h`; if dataset dimensions differ, the config dimension is overridden dynamically so downstream modules stay consistent.
