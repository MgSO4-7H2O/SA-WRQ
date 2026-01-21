#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/types.h"
#include "eval/gt.h"
#include "eval/metrics.h"
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

  std::optional<std::string> base_dataset_path;
  if (dataset_spec) {
    auto resolved = ResolveFvecsPath(*dataset_spec, "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
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

  auto searcher_res = CreateHybridSearcher(config);
  if (!searcher_res.ok()) {
    std::cerr << searcher_res.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());

  auto gt_res = ComputeGroundTruth(Q, X, config.topk);
  if (!gt_res.ok()) {
    std::cerr << gt_res.status().ToString() << std::endl;
    return 1;
  }
  auto gt = gt_res.value();

  SearchParams params;
  params.topk = config.topk;
  params.nprobe = config.nprobe;
  params.use_whitening = config.use_whitening;
  params.enable_dual_route = config.enable_dual_route;

  std::vector<std::vector<DocId>> predictions;
  predictions.reserve(nq);

  for (uint32_t i = 0; i < nq; ++i) {
    Eigen::VectorXf q = Q.row(i).transpose();
    auto search_res = searcher->Search(q, params);
    if (!search_res.ok()) {
      std::cerr << search_res.status().ToString() << std::endl;
      return 1;
    }
    std::vector<DocId> row;
    for (const auto& cand : search_res.value().topk) {
      row.push_back(cand.doc_id);
    }
    predictions.push_back(std::move(row));
  }

  auto recall_res = RecallAtK(gt, predictions, config.topk);
  if (!recall_res.ok()) {
    std::cerr << recall_res.status().ToString() << std::endl;
    return 1;
  }
  std::cout << "Recall@" << config.topk << " = " << recall_res.value() << std::endl;

  return 0;
}
