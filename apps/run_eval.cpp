#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include "common/config.h"
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

  auto searcher_res = CreateHybridSearcher(config);
  if (!searcher_res.ok()) {
    std::cerr << searcher_res.status().ToString() << std::endl;
    return 1;
  }
  std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());

  const uint32_t nx = 64;
  const uint32_t nq = 8;
  MatrixRM X = GenerateRandom(nx, config.dim, config.seed);
  MatrixRM Q = GenerateRandom(nq, config.dim, config.seed + 1);

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
