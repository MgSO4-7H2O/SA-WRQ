#include <cassert>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "common/config.h"
#include "eval/gt.h"
#include "eval/metrics.h"
#include "search/hybrid_search.h"

using namespace ann;

int main() {
  Config cfg;
  cfg.dim = 8;
  cfg.topk = 5;
  MatrixRM X = MatrixRM::Random(32, cfg.dim);
  MatrixRM Q = MatrixRM::Random(4, cfg.dim);

  auto gt_res = ComputeGroundTruth(Q, X, cfg.topk);
  assert(gt_res.ok());
  auto searcher_res = CreateHybridSearcher(cfg);
  assert(searcher_res.ok());
  std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());

  SearchParams params;
  params.topk = cfg.topk;
  params.nprobe = cfg.nprobe;
  params.use_whitening = cfg.use_whitening;
  params.enable_dual_route = cfg.enable_dual_route;

  std::vector<std::vector<DocId>> pred;
  for (int i = 0; i < Q.rows(); ++i) {
    Eigen::VectorXf q = Q.row(i).transpose();
    auto res = searcher->Search(q, params);
    assert(res.ok());
    std::vector<DocId> row;
    for (const auto& cand : res.value().topk) {
      row.push_back(cand.doc_id);
    }
    pred.push_back(std::move(row));
  }

  auto recall = RecallAtK(gt_res.value(), pred, cfg.topk);
  assert(recall.ok());
  std::cout << "Smoke recall: " << recall.value() << std::endl;
  return 0;
}
