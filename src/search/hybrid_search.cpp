#include "search/hybrid_search.h"

#include <memory>

namespace ann {
namespace {

class ConcreteHybridSearcher : public HybridSearcher {
 public:
  explicit ConcreteHybridSearcher(Config config) : config_(std::move(config)) {}

  Status SetIndex(std::shared_ptr<IVFIndex> index, const VersionSet& versions) override {
    if (!index) {
      return Status::InvalidArgument("index is null");
    }
    if (versions.index_version == 0) {
      return Status::InvalidArgument("index_version must be set");
    }
    ivf_ = std::move(index);
    route_versions_ = versions;
    return Status::OK();
  }

  Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> query,
                              const SearchParams& params) const override {
    if (static_cast<uint32_t>(query.size()) != config_.dim) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    if (params.topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    if (!ivf_) {
      return Status::Unavailable("IVF index not initialized");
    }
    auto search_res = ivf_->Search(query, params.topk, params.nprobe, route_versions_,
                                   params.enable_dual_route ? 1 : 0);
    if (!search_res.ok()) {
      return search_res.status();
    }
    SearchResult result;
    result.topk = std::move(search_res.value());
    return result;
  }

 private:
  Config config_;
  std::shared_ptr<IVFIndex> ivf_;
  VersionSet route_versions_{};
};

}  // namespace

Result<std::unique_ptr<HybridSearcher>> CreateHybridSearcher(const Config& config) {
  return std::unique_ptr<HybridSearcher>(new ConcreteHybridSearcher(config));
}

}  // namespace ann
