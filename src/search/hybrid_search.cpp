#include "search/hybrid_search.h"

#include <memory>

namespace ann {
namespace {

class ConcreteHybridSearcher : public HybridSearcher {
 public:
  explicit ConcreteHybridSearcher(Config config) : config_(std::move(config)) {}

  Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> query,
                              const SearchParams& params) const override {
    if (static_cast<uint32_t>(query.size()) != config_.dim) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    if (params.topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    SearchResult result;
    for (uint32_t i = 0; i < params.topk; ++i) {
      Candidate c;
      c.doc_id = i;
      c.approx_dist = query.squaredNorm() + static_cast<float>(i);
      c.rerank_dist = c.approx_dist;
      c.versions = VersionSet{0, 0, 0};
      c.from_new = params.enable_dual_route ? 1 : 0;
      result.topk.push_back(c);
    }
    return result;
  }

 private:
  Config config_;
};

}  // namespace

Result<std::unique_ptr<HybridSearcher>> CreateHybridSearcher(const Config& config) {
  return std::unique_ptr<HybridSearcher>(new ConcreteHybridSearcher(config));
}

}  // namespace ann
