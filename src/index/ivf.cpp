#include "index/ivf.h"

#include <algorithm>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_set>

namespace ann {
namespace {

class StubIVFIndex : public IVFIndex {
 public:
  Result<VersionId> Build(Eigen::Ref<const MatrixRM> Xw,
                          const std::vector<DocId>& ids,
                          const IVFParams& p,
                          VersionId index_version) override {
    if (Xw.rows() == 0 || Xw.cols() == 0) {
      return Status::InvalidArgument("Xw is empty");
    }
    if (ids.size() != static_cast<size_t>(Xw.rows())) {
      return Status::InvalidArgument("ids size mismatch");
    }
    if (p.nlist == 0) {
      return Status::InvalidArgument("nlist must be > 0");
    }
    std::unique_lock lock(mu_);
    dim_ = static_cast<uint32_t>(Xw.cols());
    VersionId version = (index_version == 0) ? next_version_++ : index_version;
    built_versions_.insert(version);
    return version;
  }

  Status Add(const AlignedVector<VectorRecord>& recs) override {
    std::unique_lock lock(mu_);
    for (const auto& rec : recs) {
      staging_.push_back(rec);
    }
    return Status::OK();
  }

  Result<std::vector<Candidate>> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                                        uint32_t topk,
                                        uint32_t nprobe,
                                        const VersionSet& route_versions,
                                        uint8_t from_new) const override {
    (void)nprobe;
    std::shared_lock lock(mu_);
    if (built_versions_.count(route_versions.index_version) == 0) {
      return Status::NotFound("Index version not built");
    }
    if (dim_ == 0 || static_cast<uint32_t>(qw.size()) != dim_) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    std::vector<Candidate> res;
    res.reserve(topk);
    for (const auto& rec : staging_) {
      Candidate c;
      c.doc_id = rec.doc_id;
      c.approx_dist = static_cast<float>(qw.squaredNorm());
      c.rerank_dist = c.approx_dist;
      c.versions = route_versions;
      c.from_new = from_new;
      res.push_back(c);
      if (res.size() >= topk) {
        break;
      }
    }
    return res;
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

 private:
  uint32_t dim_{0};
  VersionId next_version_{1};
  mutable std::shared_mutex mu_;
  std::unordered_set<VersionId> built_versions_;
  AlignedVector<VectorRecord> staging_;
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<StubIVFIndex>(); }

}  // namespace ann
