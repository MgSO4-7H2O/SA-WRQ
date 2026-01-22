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
# Build index workflow (optional standalone pipeline)
./build_index ../configs/base.json [base_fvecs_or_directory]

# Evaluation workflow
./run_eval ../configs/base.json [dataset_dir_or_base_fvecs] [query_fvecs]
```

- When `.fvecs` paths (FAISS format) are provided, the tools load them through `LoadFvecs` (row-major) instead of生成 toy 数据；也可以传入目录（如 `data/cifar60k`）自动寻找 `_base.fvecs` 和 `_query.fvecs`。
- `run_eval` 离线评测流程：
  1. 读取库/查询向量（真实数据或固定随机种子）。
  2. 运行 Exact Search（暴力 L2）得到标准答案，并确保 Recall@K=1 作为 sanity check。
  3. 训练真实 IVF（k-means coarse centroids + 倒排表），HybridSearcher 注入该索引并执行 nprobe 检索。
  4. 输出 IVF Recall@K、延迟分位（p50/p99）与 QPS，方便对比不同 `nprobe`/`nlist` 参数以及后续白化等模块优化。
- 两个 CLI 均接受 JSON 配置（字段在 `include/common/config.h`），当数据维度与 `dim` 不一致时 CLI 会在运行期覆盖以保持下游一致。
