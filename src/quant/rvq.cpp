#include "quant/rvq.h"

#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace ann {
namespace {

class StubRVQCodebook : public RVQCodebook {
 public:
  Result<VersionId> Train(Eigen::Ref<const MatrixRM> Xw, const RVQParams& p) override {
    if (Xw.rows() == 0 || Xw.cols() == 0) {
      return Status::InvalidArgument("Xw must be non-empty");
    }
    if (p.num_layers == 0 || p.codewords == 0) {
      return Status::InvalidArgument("RVQParams invalid");
    }
    std::unique_lock lock(mu_);
    dim_ = static_cast<uint32_t>(Xw.cols());
    p_ = p;
    VersionId version = next_version_++;
    trained_versions_.insert(version);
    return version;
  }

  Result<void> Encode(Eigen::Ref<const Eigen::VectorXf> xw,
                      VersionId version,
                      std::vector<uint32_t>* out_codes,
                      Eigen::VectorXf* out_residual) const override {
    if (out_codes == nullptr) {
      return Status::InvalidArgument("out_codes is null");
    }
    std::shared_lock lock(mu_);
    if (trained_versions_.count(version) == 0) {
      return Status::NotFound("RVQ version missing");
    }
    if (dim_ == 0 || static_cast<uint32_t>(xw.size()) != dim_) {
      return Status::InvalidArgument("Dimension mismatch in Encode");
    }
    out_codes->assign(p_.num_layers, 0u);
    if (out_residual != nullptr) {
      if (out_residual->size() != xw.size()) {
        return Status::InvalidArgument("Residual dim mismatch");
      }
      *out_residual = xw;
    }
    return Result<void>();
  }

  Result<VersionId> UpdateTail(
      Eigen::Ref<const MatrixRM> residuals,
      Eigen::Ref<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> codes,
      const RVQParams& p,
      float) override {
    if (residuals.rows() == 0 || residuals.cols() == 0) {
      return Status::InvalidArgument("Residuals empty");
    }
    if (codes.rows() != residuals.rows()) {
      return Status::InvalidArgument("Codes rows mismatch");
    }
    std::unique_lock lock(mu_);
    p_ = p;
    dim_ = static_cast<uint32_t>(residuals.cols());
    VersionId version = next_version_++;
    trained_versions_.insert(version);
    return version;
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

 private:
  mutable std::shared_mutex mu_;
  VersionId next_version_{1};
  uint32_t dim_{0};
  RVQParams p_{};
  std::unordered_set<VersionId> trained_versions_;
};

}  // namespace

std::shared_ptr<RVQCodebook> CreateRVQCodebook() {
  return std::make_shared<StubRVQCodebook>();
}

}  // namespace ann
