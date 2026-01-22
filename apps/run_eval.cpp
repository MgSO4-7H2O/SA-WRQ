#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "search/exact_search.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

MatrixRM GenerateRandom(uint32_t rows, uint32_t cols, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  MatrixRM m(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      m(r, c) = dist(gen);
    }
  }
  return m;
}

}  // namespace

struct EvalMetrics {
  bool use_whitening{false};
  double recall{0.0};
  double p50_ms{0.0};
  double p99_ms{0.0};
  double qps{0.0};
};

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> dataset_spec = (argc > 2) ? std::optional<std::string>(argv[2]) : std::nullopt;
  std::optional<std::string> query_spec = (argc > 3) ? std::optional<std::string>(argv[3]) : std::nullopt;
  std::string dataset_label = "synthetic";

  std::optional<std::string> base_dataset_path;
  if (dataset_spec) {
    auto resolved = ResolveFvecsPath(*dataset_spec, "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
    std::filesystem::path ds_path(*base_dataset_path);
    auto parent_name = ds_path.parent_path().filename().string();
    if (!parent_name.empty()) {
      dataset_label = parent_name;
    } else {
      dataset_label = ds_path.stem().string();
    }
  }

  std::optional<std::string> query_dataset_path;
  if (query_spec) {
    auto resolved_query = ResolveFvecsPath(*query_spec, "_query.fvecs");
    if (!resolved_query.ok()) {
      std::cerr << resolved_query.status().ToString() << std::endl;
      return 1;
    }
    query_dataset_path = resolved_query.value();
    std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
  } else if (dataset_spec) {
    namespace fs = std::filesystem;
    fs::path spec_path(*dataset_spec);
    std::error_code ec;
    if (fs::is_directory(spec_path, ec)) {
      auto resolved_query = ResolveFvecsPath(*dataset_spec, "_query.fvecs");
      if (resolved_query.ok()) {
        query_dataset_path = resolved_query.value();
        std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
      } else {
        std::cout << "[WARN] " << resolved_query.status().ToString()
                  << ". Falling back to random queries." << std::endl;
      }
    } else if (base_dataset_path) {
      auto parent = fs::path(*base_dataset_path).parent_path();
      if (!parent.empty()) {
        auto resolved_query = ResolveFvecsPath(parent.string(), "_query.fvecs");
        if (resolved_query.ok()) {
          query_dataset_path = resolved_query.value();
          std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
        }
      }
    } else {
      dataset_label = spec_path.filename().string();
    }
  }

  uint32_t nx = 0;
  MatrixRM X;
  if (base_dataset_path) {
    auto load_res = LoadFvecs(*base_dataset_path);
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    X = load_res.value();
    nx = static_cast<uint32_t>(X.rows());
    if (nx == 0) {
      std::cerr << "Base dataset contains no vectors." << std::endl;
      return 1;
    }
    if (config.dim != static_cast<uint32_t>(X.cols())) {
      std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << X.cols() << std::endl;
      config.dim = static_cast<uint32_t>(X.cols());
    }
  } else {
    nx = 64;
    X = GenerateRandom(nx, config.dim, config.seed);
  }

  uint32_t nq = 0;
  MatrixRM Q;
  if (query_dataset_path) {
    auto load_res = LoadFvecs(*query_dataset_path);
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    Q = load_res.value();
    nq = static_cast<uint32_t>(Q.rows());
    if (nq == 0) {
      std::cerr << "Query dataset contains no vectors." << std::endl;
      return 1;
    }
    if (static_cast<uint32_t>(Q.cols()) != config.dim) {
      if (!base_dataset_path) {
        std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << Q.cols() << std::endl;
        config.dim = static_cast<uint32_t>(Q.cols());
      } else {
        std::cerr << "Query dimension " << Q.cols() << " mismatches base " << config.dim << std::endl;
        return 1;
      }
    }
  } else {
    nq = 8;
    Q = GenerateRandom(nq, config.dim, config.seed + 1);
  }

  auto exact_res = ExactSearchBatch(Q, X, config.topk);
  if (!exact_res.ok()) {
    std::cerr << exact_res.status().ToString() << std::endl;
    return 1;
  }
  auto ground_truth_plain = exact_res.value();
  auto sanity = RecallAtK(ground_truth_plain, ground_truth_plain, config.topk);
  if (!sanity.ok() || sanity.value() != 1.0f) {
    std::cerr << "Exact search sanity failed" << std::endl;
    return 1;
  }

  std::vector<DocId> ids(nx);
  std::iota(ids.begin(), ids.end(), 0);

  SearchParams base_params;
  base_params.topk = config.topk;
  base_params.nprobe = config.nprobe;
  base_params.use_whitening = false;
  base_params.enable_dual_route = config.enable_dual_route;

  auto run_experiment = [&](bool use_whitening) -> Result<EvalMetrics> {
    auto ivf = CreateIVFIndex();
    auto searcher_res = CreateHybridSearcher(config);
    if (!searcher_res.ok()) {
      return searcher_res.status();
    }
    std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());
    MatrixRM X_index = X;
    MatrixRM Q_input = Q;
    VersionId whiten_version = 0;
    std::shared_ptr<WhiteningModel> whitening;
    std::vector<std::vector<DocId>> local_gt;
    const std::vector<std::vector<DocId>>* gt_ptr = &ground_truth_plain;

    if (use_whitening) {
      whitening = CreateWhiteningModel();
      auto version_res = whitening->Fit(X);
      if (!version_res.ok()) {
        return version_res.status();
      }
      whiten_version = version_res.value();
      auto base_batch = whitening->TransformBatch(X, whiten_version);
      if (!base_batch.ok()) {
        return base_batch.status();
      }
      X_index = base_batch.value();
      auto query_batch = whitening->TransformBatch(Q, whiten_version);
      if (!query_batch.ok()) {
        return query_batch.status();
      }
      Q_input = query_batch.value();
      auto gt_res = ExactSearchBatch(Q_input, X_index, config.topk);
      if (!gt_res.ok()) {
        return gt_res.status();
      }
      local_gt = gt_res.value();
      gt_ptr = &local_gt;
    }

    IVFParams ivf_params;
    ivf_params.nlist = std::max(1u, config.ivf_nlist);
    ivf_params.dim = config.dim;
    auto ivf_version_res = ivf->Build(X_index, ids, ivf_params, 0);
    if (!ivf_version_res.ok()) {
      return ivf_version_res.status();
    }
    VersionSet versions{whiten_version, 0, ivf_version_res.value()};
    Status index_status = searcher->SetIndex(ivf, versions);
    if (!index_status.ok()) {
      return index_status;
    }
    if (use_whitening) {
      Status wstatus = searcher->SetWhitening(whitening, whiten_version);
      if (!wstatus.ok()) {
        return wstatus;
      }
    } else {
      searcher->SetWhitening(nullptr, 0);
    }

    SearchParams params = base_params;
    params.use_whitening = use_whitening;

    std::vector<std::vector<DocId>> predictions(nq);
    std::vector<double> latencies_ms(nq, 0.0);
    std::atomic<bool> failed{false};
    std::mutex error_mu;
    Status error_status;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(nq); ++i) {
      if (failed.load()) {
        continue;
      }
      Eigen::VectorXf qvec = Q.row(i).transpose();
      Timer timer;
      auto search_res = searcher->Search(qvec, params);
      double elapsed = timer.ElapsedMillis();
      if (!search_res.ok()) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!failed.exchange(true)) {
          error_status = search_res.status();
        }
        continue;
      }
      std::vector<DocId> row;
      row.reserve(search_res.value().topk.size());
      for (const auto& cand : search_res.value().topk) {
        row.push_back(cand.doc_id);
      }
      predictions[i] = std::move(row);
      latencies_ms[i] = elapsed;
    }

    if (failed.load()) {
      return error_status;
    }

    auto recall_res = RecallAtK(*gt_ptr, predictions, config.topk);
    if (!recall_res.ok()) {
      return recall_res.status();
    }

    double total_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    double qps = (total_ms > 0.0) ? (static_cast<double>(nq) / (total_ms / 1000.0)) : 0.0;
    auto summary = SummarizeLatencies(latencies_ms);
    if (!summary.ok()) {
      return summary.status();
    }

    EvalMetrics metrics;
    metrics.use_whitening = use_whitening;
    metrics.recall = recall_res.value();
    metrics.p50_ms = summary.value().p50_ms;
    metrics.p99_ms = summary.value().p99_ms;
    metrics.qps = qps;
    return metrics;
  };

  std::cout << "Exact Recall@" << config.topk << " = 1.0" << std::endl;

  auto save_metrics = [&](const EvalMetrics& metrics) {
    std::filesystem::path results_dir = std::filesystem::path("result") / dataset_label;
    std::error_code ec;
    std::filesystem::create_directories(results_dir, ec);
    const auto now = std::chrono::system_clock::now();
    const auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string suffix = metrics.use_whitening ? "zca" : "plain";
    std::string file_name = "topk" + std::to_string(config.topk) + "_np" +
                            std::to_string(base_params.nprobe) + "_nl" +
                            std::to_string(config.ivf_nlist) + "_" + suffix + "_" + std::to_string(ts) + ".json";
    std::filesystem::path result_path = results_dir / file_name;
    std::ofstream ofs(result_path);
    if (ofs) {
      ofs << "{\n";
      ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
      ofs << "  \"timestamp\": " << ts << ",\n";
      ofs << "  \"use_whitening\": " << (metrics.use_whitening ? "true" : "false") << ",\n";
      ofs << "  \"metrics\": {\n";
      ofs << "    \"recall@"
          << config.topk << "\": " << metrics.recall << ",\n";
      ofs << "    \"latency_ms\": {\"p50\": " << metrics.p50_ms << ", \"p99\": " << metrics.p99_ms << "},\n";
      ofs << "    \"qps\": " << metrics.qps << "\n";
      ofs << "  },\n";
      ofs << "  \"params\": {\n";
      ofs << "    \"topk\": " << config.topk << ",\n";
      ofs << "    \"nprobe\": " << base_params.nprobe << ",\n";
      ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
      ofs << "    \"rvq_layers\": " << config.rvq_layers << ",\n";
      ofs << "    \"rvq_codewords\": " << config.rvq_codewords << ",\n";
      ofs << "    \"use_whitening\": " << (metrics.use_whitening ? "true" : "false") << "\n";
      ofs << "  }\n";
      ofs << "}\n";
      std::cout << "Saved metrics to " << result_path << std::endl;
    } else {
      std::cerr << "Failed to write results to " << result_path << std::endl;
    }
  };

  std::vector<bool> modes = {false};
  if (config.use_whitening) {
    modes.push_back(true);
  }

  for (bool mode : modes) {
    auto metrics_res = run_experiment(mode);
    if (!metrics_res.ok()) {
      std::cerr << metrics_res.status().ToString() << std::endl;
      if (!mode) {
        return 1;
      }
      continue;
    }
    const EvalMetrics& metrics = metrics_res.value();
    std::cout << (mode ? "[IVF+ZCA] " : "[IVF] ") << "Recall@" << config.topk << " = " << metrics.recall
              << " (nprobe=" << base_params.nprobe << ")" << std::endl;
    std::cout << "Latency p50=" << metrics.p50_ms << "ms, p99=" << metrics.p99_ms << "ms, QPS=" << metrics.qps
              << std::endl;
    save_metrics(metrics);
  }

  return 0;
}
