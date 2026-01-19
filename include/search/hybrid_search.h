#pragma once

#include <memory>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"
#include "common/config.h"

namespace ann {

struct SearchParams {
  uint32_t topk{10};
  uint32_t nprobe{8};
  bool use_whitening{true};
  bool enable_dual_route{true};
};

class HybridSearcher {
 public:
  virtual ~HybridSearcher() = default;

  // TS: Executes retrieval given a single query vector.
  virtual Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> query,
                                      const SearchParams& params) const = 0;
};

Result<std::unique_ptr<HybridSearcher>> CreateHybridSearcher(const Config& config);

}  // namespace ann
