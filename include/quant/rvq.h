#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct RVQParams {
  uint32_t num_layers{2};
  uint32_t codewords{256};
  bool update_tail_only{false};
  uint32_t stable_prefix_layers{0};
};

class RVQCodebook {
 public:
  virtual ~RVQCodebook() = default;

  // NTS: Trains a fresh version from whitened vectors (Xw shape: n x dim).
  virtual Result<VersionId> Train(Eigen::Ref<const MatrixRM> Xw, const RVQParams& p) = 0;

  // TS: Encodes a single whitened vector using a fixed version.
  virtual Result<void> Encode(Eigen::Ref<const Eigen::VectorXf> xw,
                              VersionId version,
                              std::vector<uint32_t>* out_codes,
                              Eigen::VectorXf* out_residual) const = 0;

  // NTS: Updates only the tail layers using residuals + codes snapshot.
  virtual Result<VersionId> UpdateTail(
      Eigen::Ref<const MatrixRM> residuals,
      Eigen::Ref<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> codes,
      const RVQParams& p,
      float ema_lr) = 0;

  virtual Result<std::vector<uint8_t>> Serialize() const = 0;
  virtual Status Deserialize(const std::vector<uint8_t>& bytes) = 0;
};

std::shared_ptr<RVQCodebook> CreateRVQCodebook();

}  // namespace ann
