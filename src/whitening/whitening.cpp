#include "whitening/whitening.h"

#include <mutex>
#include <random>
#include <unordered_map>

namespace ann {
namespace {

class StubWhiteningModel : public WhiteningModel {
 public:
  Result<VersionId> Fit(Eigen::Ref<const MatrixRM> X) override {
    if (X.rows() == 0 || X.cols() == 0) {
      return Status::InvalidArgument("Input matrix must be non-empty");
    }
    std::unique_lock lock(mu_);
    dim_ = static_cast<uint32_t>(X.cols());
    pending_stats_ = X.colwise().mean();
    mean_ = pending_stats_;
    pending_version_ = next_version_++;
    transforms_[pending_version_] = MatrixRM::Identity(dim_, dim_);
    return pending_version_;
  }

  Result<void> Transform(Eigen::Ref<const Eigen::VectorXf> x,
                         VersionId version,
                         Eigen::Ref<Eigen::VectorXf> out_xw) const override {
    std::shared_lock lock(mu_);
    if (dim_ == 0) {
      return Status::Unavailable("Whitening model not initialized");
    }
    if (static_cast<uint32_t>(x.size()) != dim_ || out_xw.size() != x.size()) {
      return Status::InvalidArgument("Dimension mismatch for Transform");
    }
    auto it = transforms_.find(version);
    if (it == transforms_.end()) {
      return Status::NotFound("Whitening version missing");
    }
    out_xw = x - mean_;
    return Result<void>();
  }

  Status UpdateStats(Eigen::Ref<const Eigen::VectorXf> x) override {
    std::unique_lock lock(mu_);
    if (dim_ == 0) {
      dim_ = static_cast<uint32_t>(x.size());
      mean_ = Eigen::VectorXf::Zero(x.size());
    }
    if (static_cast<uint32_t>(x.size()) != dim_) {
      return Status::InvalidArgument("Dim mismatch in UpdateStats");
    }
    mean_ = 0.9f * mean_ + 0.1f * x;
    pending_version_ = next_version_;
    return Status::OK();
  }

  Result<VersionId> FinalizeNewVersion() override {
    std::unique_lock lock(mu_);
    if (pending_version_ == 0) {
      return Status::InvalidArgument("No stats pending");
    }
    VersionId v = pending_version_;
    transforms_[v] = MatrixRM::Identity(dim_, dim_);
    mean_ = pending_stats_;
    pending_version_ = 0;
    return v;
  }

  Result<MatrixRM> Bridge(VersionId src_version, VersionId dst_version) const override {
    std::shared_lock lock(mu_);
    if (transforms_.count(src_version) == 0 || transforms_.count(dst_version) == 0) {
      return Status::NotFound("Version not found for bridge");
    }
    MatrixRM bridge = MatrixRM::Identity(dim_, dim_);
    return bridge;
  }

  Result<std::vector<uint8_t>> Serialize() const override {
    return std::vector<uint8_t>{};
  }

  Status Deserialize(const std::vector<uint8_t>&) override {
    return Status::OK();
  }

 private:
  mutable std::shared_mutex mu_;
  uint32_t dim_{0};
  VersionId next_version_{1};
  VersionId pending_version_{0};
  Eigen::VectorXf mean_{Eigen::VectorXf()};
  Eigen::VectorXf pending_stats_{Eigen::VectorXf()};
  std::unordered_map<VersionId, MatrixRM> transforms_;
};

}  // namespace

std::shared_ptr<WhiteningModel> CreateWhiteningModel() {
  return std::make_shared<StubWhiteningModel>();
}

}  // namespace ann
