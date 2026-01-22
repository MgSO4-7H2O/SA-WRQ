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

  std::vector<DocId> ids(nx);
  std::iota(ids.begin(), ids.end(), 0);

  auto ivf = CreateIVFIndex();
  IVFParams ivf_params;
  ivf_params.nlist = std::max(1u, config.ivf_nlist);
  ivf_params.dim = config.dim;
  auto ivf_version_res = ivf->Build(X, ids, ivf_params, 0);
  if (!ivf_version_res.ok()) {
    std::cerr << ivf_version_res.status().ToString() << std::endl;
    return 1;
  }
  VersionId ivf_version = ivf_version_res.value();

  auto searcher_res = CreateHybridSearcher(config);
  if (!searcher_res.ok()) {
    std::cerr << searcher_res.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());
  VersionSet route_versions{0, 0, ivf_version};
  Status set_status = searcher->SetIndex(ivf, route_versions);
  if (!set_status.ok()) {
    std::cerr << set_status.ToString() << std::endl;
    return 1;
  }

  auto exact_res = ExactSearchBatch(Q, X, config.topk);
  if (!exact_res.ok()) {
    std::cerr << exact_res.status().ToString() << std::endl;
    return 1;
  }
  auto ground_truth = exact_res.value();
  auto sanity = RecallAtK(ground_truth, ground_truth, config.topk);
  if (!sanity.ok() || sanity.value() != 1.0f) {
    std::cerr << "Exact search sanity failed" << std::endl;
    return 1;
  }

  SearchParams params;
  params.topk = config.topk;
  params.nprobe = config.nprobe;
  params.use_whitening = config.use_whitening;
  params.enable_dual_route = config.enable_dual_route;

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
    std::cerr << error_status.ToString() << std::endl;
    return 1;
  }

  auto recall_res = RecallAtK(ground_truth, predictions, config.topk);
  if (!recall_res.ok()) {
    std::cerr << recall_res.status().ToString() << std::endl;
    return 1;
  }

  double total_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
  double qps = (total_ms > 0.0) ? (static_cast<double>(nq) / (total_ms / 1000.0)) : 0.0;
  auto summary = SummarizeLatencies(latencies_ms);
  if (!summary.ok()) {
    std::cerr << summary.status().ToString() << std::endl;
    return 1;
  }

  const double recall_value = recall_res.value();
  const double p50 = summary.value().p50_ms;
  const double p99 = summary.value().p99_ms;
  std::cout << "Exact Recall@" << config.topk << " = 1.0" << std::endl;
  std::cout << "IVF Recall@" << config.topk << " = " << recall_value << " (nprobe=" << params.nprobe
            << ")" << std::endl;
  std::cout << "Latency p50=" << p50 << "ms, p99=" << p99 << "ms, QPS=" << qps << std::endl;

  if (dataset_spec) {
    std::filesystem::path label_path(*dataset_spec);
    if (std::filesystem::is_directory(label_path)) {
      auto name = label_path.filename().string();
      if (!name.empty()) {
        dataset_label = name;
      }
    } else if (!base_dataset_path) {
      dataset_label = label_path.parent_path().filename().string();
      if (dataset_label.empty()) {
        dataset_label = label_path.stem().string();
      }
    }
  }
  if (dataset_label.empty()) {
    dataset_label = "synthetic";
  }

  std::filesystem::path results_dir = std::filesystem::path("result") / dataset_label;
  std::error_code ec;
  std::filesystem::create_directories(results_dir, ec);
  const auto now = std::chrono::system_clock::now();
  const auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::string file_name = "topk" + std::to_string(config.topk) + "_np" + std::to_string(params.nprobe) +
                          "_nl" + std::to_string(config.ivf_nlist) + "_" + std::to_string(ts) + ".json";
  std::filesystem::path result_path = results_dir / file_name;
  std::ofstream ofs(result_path);
  if (ofs) {
    ofs << "{\n";
    ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
    ofs << "  \"timestamp\": " << ts << ",\n";
    ofs << "  \"metrics\": {\n";
    ofs << "    \"recall@"
        << config.topk << "\": " << recall_value << ",\n";
    ofs << "    \"latency_ms\": {\"p50\": " << p50 << ", \"p99\": " << p99 << "},\n";
    ofs << "    \"qps\": " << qps << "\n";
    ofs << "  },\n";
    ofs << "  \"params\": {\n";
    ofs << "    \"topk\": " << config.topk << ",\n";
    ofs << "    \"nprobe\": " << params.nprobe << ",\n";
    ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
    ofs << "    \"rvq_layers\": " << config.rvq_layers << ",\n";
    ofs << "    \"rvq_codewords\": " << config.rvq_codewords << ",\n";
    ofs << "    \"use_whitening\": " << (config.use_whitening ? "true" : "false") << "\n";
    ofs << "  }\n";
    ofs << "}\n";
    std::cout << "Saved metrics to " << result_path << std::endl;
  } else {
    std::cerr << "Failed to write results to " << result_path << std::endl;
  }

  return 0;
}
