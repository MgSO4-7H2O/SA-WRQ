#include "quant/rvq.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace ann {
namespace {

constexpr int kKMeansIterations = 15; // 训练时的迭代次数
constexpr uint32_t kDefaultSeed = 42;

// --- K-Means 辅助函数 (与 IVF 独立，可后续修改) ---

// 计算两个向量的平方欧氏距离
float SquaredDist(Eigen::Ref<const Eigen::VectorXf> a, Eigen::Ref<const Eigen::VectorXf> b) {
  return (a - b).squaredNorm();
}

// 在给定的中心点列表中，找到离 vec 最近的那个，返回 index
int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const MatrixRM& centroids) {
  float best_dist = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < centroids.rows(); ++i) {
    float d = SquaredDist(vec, centroids.row(i).transpose());
    if (d < best_dist) {
      best_dist = d;
      best_idx = i;
    }
  }
  return best_idx;
}

// 运行 K-Means 聚类
// input: 输入数据 (N x Dim)
// k: 聚类中心数量
// seed: 随机种子
// 返回: 聚类中心矩阵 (K x Dim)
MatrixRM RunKMeansLocal(Eigen::Ref<const MatrixRM> input, int k, uint32_t seed) {
  const int64_t n = input.rows();
  const int64_t dim = input.cols();
  
  // 1. 初始化中心点 (随机采样)
  MatrixRM centroids(k, dim);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int64_t> dist(0, n - 1);
  for (int i = 0; i < k; ++i) {
    centroids.row(i) = input.row(dist(gen));
  }

  std::vector<int> assignments(n);
  
  // 2. 迭代优化
  for (int iter = 0; iter < kKMeansIterations; ++iter) {
    // E-Step: 分配簇
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < n; ++i) {
      assignments[i] = NearestCentroid(input.row(i).transpose(), centroids);
    }

    // M-Step: 更新中心
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int> counts(k, 0);
    
    for (int64_t i = 0; i < n; ++i) {
      int c = assignments[i];
      new_centroids.row(c) += input.row(i);
      counts[c]++;
    }

    // 处理空簇并归一化
    for (int c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        new_centroids.row(c) /= static_cast<float>(counts[c]);
      } else {
        // 如果某个簇是空的，随机再选一个点作为新中心
        new_centroids.row(c) = input.row(dist(gen));
      }
    }
    centroids = new_centroids;
  }
  return centroids;
}

// --- RVQ 实现 ---

struct RVQModel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VersionId version{0};
  uint32_t dim{0};
  RVQParams params;
  // codebooks[layer_id] 是一个 (Codewords x Dim) 的矩阵
  std::vector<MatrixRM> codebooks; 
};

class KMeansRVQCodebook : public RVQCodebook {
 public:
  Result<VersionId> Train(Eigen::Ref<const MatrixRM> Xw, const RVQParams& p) override {
    if (Xw.rows() == 0 || Xw.cols() == 0) {
      return Status::InvalidArgument("Xw must be non-empty");
    }
    if (p.num_layers == 0 || p.codewords == 0) {
      return Status::InvalidArgument("RVQParams invalid");
    }

    std::cout << "[RVQ] Training started. Layers=" << p.num_layers 
              << ", Codewords=" << p.codewords << std::endl;

    auto model = std::make_unique<RVQModel>();
    model->dim = static_cast<uint32_t>(Xw.cols());
    model->params = p;
    model->codebooks.resize(p.num_layers);

    // residual_data 初始就是原始数据 Xw
    // 用 MatrixRM 存储，因为 K-Means 也要用 MatrixRM
    MatrixRM current_residuals = Xw; 

    for (uint32_t l = 0; l < p.num_layers; ++l) {
      std::cout << "[RVQ] Training Layer " << l + 1 << "..." << std::endl;
      
      // 1. 对当前的残差进行聚类
      MatrixRM centroids = RunKMeansLocal(current_residuals, p.codewords, 42 + l);
      model->codebooks[l] = centroids;

      // 2. 计算新的残差 (current_residuals = current_residuals - nearest_centroid)
      // 给下一层准备数据
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int64_t i = 0; i < current_residuals.rows(); ++i) {
        Eigen::VectorXf vec = current_residuals.row(i).transpose();
        int best_idx = NearestCentroid(vec, centroids);
        // 更新残差：原残差 - 中心点
        Eigen::VectorXf new_res = vec - centroids.row(best_idx).transpose();
        current_residuals.row(i) = new_res;
      }
    }

    std::unique_lock lock(mu_);
    VersionId version = next_version_++;
    model->version = version;
    models_[version] = std::move(model);
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
    auto it = models_.find(version);
    if (it == models_.end()) {
      return Status::NotFound("RVQ version missing");
    }
    const auto& model = *it->second;

    if (static_cast<uint32_t>(xw.size()) != model.dim) {
      return Status::InvalidArgument("Dimension mismatch in Encode");
    }

    out_codes->resize(model.params.num_layers);
    
    // 在循环中会不断修改 vector 来逼近真实值
    Eigen::VectorXf current_vec = xw;

    for (uint32_t l = 0; l < model.params.num_layers; ++l) {
      const MatrixRM& centroids = model.codebooks[l];
      
      // 1. 找最近的中心点
      int code = NearestCentroid(current_vec, centroids);
      (*out_codes)[l] = static_cast<uint32_t>(code);

      // 2. 减去中心点，得到残差
      current_vec -= centroids.row(code).transpose();
    }

    if (out_residual != nullptr) {
      *out_residual = current_vec; // 最后的 current_vec 就是经过所有层后的最终残差
    }
    return Result<void>();
  }

  // --- TODO ---
  Result<VersionId> UpdateTail(
      Eigen::Ref<const MatrixRM> residuals,
      Eigen::Ref<const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> codes,
      const RVQParams& p,
      float) override {
      // 占位，在线更新TODO
      return Status::Unimplemented("UpdateTail not implemented yet");
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }
  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

  Result<MatrixRM> ComputeInnerProductTable(Eigen::Ref<const Eigen::VectorXf> query, VersionId version) const override {
    std::shared_lock lock(mu_);
    auto it = models_.find(version);
    if (it == models_.end()) {
        return Status::NotFound("RVQ version missing for LUT computation");
    }
    const auto& model = *it->second;

    // 表大小: Layers x Codewords
    MatrixRM lut(model.params.num_layers, model.params.codewords);
    
    // 逐层计算 query 与该层所有 centroids 的内积
    for (uint32_t l = 0; l < model.params.num_layers; ++l) {
        // model.codebooks[l] 是 (Codewords x Dim)
        // lut.row(l) = query * codebooks[l].transpose()
        // 利用 Eigen 矩阵乘法加速: (1 x D) * (D x C) -> (1 x C)
        lut.row(l) = (model.codebooks[l] * query).transpose();
    }
    return lut;
  }
 private:
  mutable std::shared_mutex mu_;
  VersionId next_version_{1};
  std::unordered_map<VersionId, std::unique_ptr<RVQModel>> models_;
};

}  // namespace

std::shared_ptr<RVQCodebook> CreateRVQCodebook() {
  return std::make_shared<KMeansRVQCodebook>();
}

}  // namespace ann
