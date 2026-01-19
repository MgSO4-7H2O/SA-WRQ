#pragma once

#include <shared_mutex>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct DriftStats {
  double residual_energy{0.0};
  double code_usage_entropy{0.0};
  double avg_margin{0.0};
};

class DriftMonitor {
 public:
  DriftMonitor();
  ~DriftMonitor();

  // NTS: Called when residual batches are produced.
  Status ObserveResidual(Eigen::Ref<const MatrixRM> residuals);

  // NTS: Called when rerank margins are observed.
  Status ObserveSearchSignal(Eigen::Ref<const Eigen::VectorXf> margins);

  // TS: Thread-safe read of trigger state.
  Result<bool> ShouldUpdateWhiten() const;
  Result<bool> ShouldUpdateTail() const;

 private:
  mutable std::shared_mutex mu_;
  DriftStats stats_{};
  bool request_whiten_{false};
  bool request_tail_{false};
};

}  // namespace ann
