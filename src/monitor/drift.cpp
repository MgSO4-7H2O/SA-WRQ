#include "monitor/drift.h"

#include <mutex>
#include <numeric>

namespace ann {

DriftMonitor::DriftMonitor() = default;
DriftMonitor::~DriftMonitor() = default;

Status DriftMonitor::ObserveResidual(Eigen::Ref<const MatrixRM> residuals) {
  if (residuals.size() == 0) {
    return Status::InvalidArgument("Residuals empty");
  }
  std::unique_lock lock(mu_);
  double energy = 0.0;
  for (int i = 0; i < residuals.rows(); ++i) {
    energy += residuals.row(i).squaredNorm();
  }
  stats_.residual_energy = 0.9 * stats_.residual_energy + 0.1 * energy;
  request_whiten_ = stats_.residual_energy > 1e4;
  return Status::OK();
}

Status DriftMonitor::ObserveSearchSignal(Eigen::Ref<const Eigen::VectorXf> margins) {
  if (margins.size() == 0) {
    return Status::InvalidArgument("Margins empty");
  }
  std::unique_lock lock(mu_);
  stats_.avg_margin = 0.9 * stats_.avg_margin + 0.1 * margins.mean();
  request_tail_ = stats_.avg_margin < 0.1;
  return Status::OK();
}

Result<bool> DriftMonitor::ShouldUpdateWhiten() const {
  std::shared_lock lock(mu_);
  return request_whiten_;
}

Result<bool> DriftMonitor::ShouldUpdateTail() const {
  std::shared_lock lock(mu_);
  return request_tail_;
}

}  // namespace ann
