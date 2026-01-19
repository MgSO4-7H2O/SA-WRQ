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
./build_index ../configs/base.json

# Evaluation workflow
./run_eval ../configs/base.json
```

Both binaries accept a path to a JSON config file matching the fields defined in `include/common/config.h`.
