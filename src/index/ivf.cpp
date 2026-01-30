#include "index/ivf.h"

#include <algorithm>
#include <cstdint>
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

constexpr int kKMeansIterations = 20;
constexpr uint32_t kDefaultSeed = 42;

struct ListEntry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DocId doc_id;
  VersionSet versions;
  Eigen::VectorXf vector;      // 存储原始向量，用于Plain测试
  std::vector<uint8_t> codes;  // 存储量化后的编码，用于RVQ测试
  float norm{0.0f};
};

struct IndexData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint32_t dim{0};
  uint32_t nlist{0};
  VersionId version{0};
  MatrixRM centroids;
  std::vector<AlignedVector<ListEntry>> lists;
};

// 辅助函数：找最近的中心点
int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const MatrixRM& centroids) {
  float best = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < centroids.rows(); ++i) {
    float dist = (centroids.row(i).transpose() - vec).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

// 重载版本，兼容旧接口
int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const IndexData& data) {
    return NearestCentroid(vec, data.centroids);
}

MatrixRM InitializeCentroids(Eigen::Ref<const MatrixRM> X, uint32_t nlist) {
  std::cout << "[IVF] Initializing " << nlist << " centroids..." << std::endl;
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  MatrixRM centroids(nlist, dim);
  std::vector<int64_t> indices(num_vecs);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen(kDefaultSeed);
  std::shuffle(indices.begin(), indices.end(), gen);
  for (uint32_t i = 0; i < nlist; ++i) {
    centroids.row(i) = X.row(indices[i % num_vecs]);
  }
  return centroids;
}

void RunKMeans(Eigen::Ref<const MatrixRM> X, MatrixRM* centroids) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(num_vecs, 0);

  std::cout << "[IVF] Starting KMeans clustering (K=" << k << ", N=" << num_vecs << ")" << std::endl;

  for (int iter = 0; iter < kKMeansIterations; ++iter) {
    if (iter % 5 == 0 || iter == kKMeansIterations - 1) {
        std::cout << "[IVF] KMeans iteration " << iter + 1 << "/" << kKMeansIterations << "..." << std::endl;
    }

    // Assignment step: 并行化！
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < num_vecs; ++i) {
      Eigen::VectorXf vec = X.row(i).transpose();
      // 这里内联最近邻搜索以减少函数调用开销，或者直接调用辅助函数
      float best = std::numeric_limits<float>::max();
      int best_idx = 0;
      for (int64_t c = 0; c < k; ++c) {
        float dist = (centroids->row(c).transpose() - vec).squaredNorm();
        if (dist < best) {
          best = dist;
          best_idx = static_cast<int>(c);
        }
      }
      assignments[i] = best_idx;
    }

    // Update step: 计算新中心
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(k, 0);
    
    // 这一步比较快，可以用简单的串行累加，或者更复杂的并行归约
    // 为了简单且安全，这里保持串行，因为 K 相对 N 较小
    for (int64_t i = 0; i < num_vecs; ++i) {
      int cluster_id = assignments[i];
      new_centroids.row(cluster_id) += X.row(i);
      counts[cluster_id]++;
    }

    std::mt19937 gen(kDefaultSeed + iter);
    std::uniform_int_distribution<int64_t> dist_index(0, num_vecs - 1);
    
    for (int64_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        new_centroids.row(c) /= static_cast<float>(counts[c]);
      } else {
        // 重新初始化空簇
        int64_t repl = dist_index(gen);
        new_centroids.row(c) = X.row(repl);
      }
    }
    *centroids = new_centroids;
  }
}

class KMeansIVFIndex : public IVFIndex {
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
      return Status::InvalidArgument("nlist must be >0");
    }

    uint32_t dim = static_cast<uint32_t>(Xw.cols());
    const uint32_t nlist = std::min<uint32_t>(p.nlist, static_cast<uint32_t>(Xw.rows()));

    MatrixRM centroids = InitializeCentroids(Xw, nlist);
    RunKMeans(Xw, &centroids);

    auto data = std::make_unique<IndexData>();
    data->dim = dim;
    data->nlist = nlist;
    data->centroids = std::move(centroids);
    data->lists.resize(nlist);

    std::cout << "[IVF] Assigning " << Xw.rows() << " vectors to inverted lists..." << std::endl;

    // 1. 并行计算归属 (Assign vectors to lists)
    std::vector<int> assignments(Xw.rows());
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < Xw.rows(); ++i) {
        assignments[i] = NearestCentroid(Xw.row(i).transpose(), data->centroids);
    }

    // 2. 串行填充列表
    for (int64_t i = 0; i < Xw.rows(); ++i) {
      int best_idx = assignments[i];
      ListEntry entry;
      entry.doc_id = ids[i];
      entry.versions = VersionSet{0, 0, 0};
      
      // 遍历原始向量，进行RVQ编码
      Eigen::VectorXf vec = Xw.row(i).transpose();
      // entry.norm = vec.squaredNorm(); // 计算模长，用于search时的距离矫正

      // 根据是否有量化器分类讨论
      if (quantizer_) {
        // RVQ
        std::vector<uint32_t> temp_codes;
        Eigen::VectorXf residual;
        quantizer_->Encode(vec, quantizer_ver_, &temp_codes, &residual);
        entry.norm = (vec - residual).squaredNorm();

        // 压缩存储，转换为uint8
        entry.codes.resize(temp_codes.size());
        for (size_t ci = 0; ci < temp_codes.size(); ++ci) {
          entry.codes[ci] = static_cast<uint8_t>(temp_codes[ci]);
        }

        // 释放vecor内存
        entry.vector.resize(0);
      } else {
        // Plain，直接存原始向量
        entry.norm = vec.squaredNorm();
        entry.vector = std::move(vec);
      }

      data->lists[best_idx].push_back(std::move(entry));
    }

    std::unique_lock lock(mu_);
    VersionId version = (index_version == 0) ? next_version_++ : index_version;
    data->version = version;
    for (auto& list : data->lists) {
      for (auto& entry : list) {
        entry.versions.index_version = version;
      }
    }
    data_map_[version] = std::move(data);
    latest_version_ = version;
    return version;
  }

  Status Add(const AlignedVector<VectorRecord>& recs) override {
    std::unique_lock lock(mu_);
    if (latest_version_ == 0) {
      return Status::InvalidArgument("Index not built");
    }
    auto it = data_map_.find(latest_version_);
    if (it == data_map_.end()) {
      return Status::NotFound("Latest version missing");
    }

    IndexData& data = *it->second;
    for (const auto& rec : recs) {
      if (rec.x.size() != data.dim) {
        return Status::InvalidArgument("Record dim mismatch");
      }
      // 寻找最近聚类
      int centroid = NearestCentroid(rec.x, data);
      ListEntry entry;
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      if (quantizer_) {
        // RVQ量化编码
        Eigen::VectorXf residual;
        std::vector<uint32_t> temp_codes;
        quantizer_->Encode(rec.x, quantizer_ver_, &temp_codes, &residual);
        entry.norm = (rec.x - residual).squaredNorm();

        // 压缩转存codes
        entry.codes.resize(temp_codes.size());
        for (size_t ci = 0; ci < temp_codes.size(); ++ci) {
          entry.codes[ci] = static_cast<uint8_t>(temp_codes[ci]);
        }
      } else {
        entry.vector = rec.x;
        entry.norm = rec.x.squaredNorm();
      }

      data.lists[centroid].push_back(std::move(entry));
    }
    return Status::OK();
  }

  Result<std::vector<Candidate>> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                                        uint32_t topk,
                                        uint32_t nprobe,
                                        const VersionSet& route_versions,
                                        uint8_t from_new) const override {
    if (topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    std::shared_lock lock(mu_);

    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;

    // 寻找最近的nprobe个聚类，并排序
    if (static_cast<uint32_t>(qw.size()) != data.dim) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    const uint32_t probes = std::max<uint32_t>(1, std::min<uint32_t>(nprobe, data.nlist));
    std::vector<std::pair<float, uint32_t>> centroid_dists(data.nlist);
    for (uint32_t i = 0; i < data.nlist; ++i) {
      float dist = (data.centroids.row(i).transpose() - qw).squaredNorm();
      centroid_dists[i] = {dist, i};
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + probes, centroid_dists.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    bool use_rvq = (quantizer_ != nullptr);

    // 预计算LUT
    MatrixRM lut;
    if (use_rvq) {
      auto lut_res = quantizer_->ComputeInnerProductTable(qw, quantizer_ver_);
      lut = lut_res.value();
    }
    
    std::vector<Candidate> candidates;
    const float qnorm = qw.squaredNorm();

    // 遍历桶内数据
    for (uint32_t pi = 0; pi < probes; ++pi) {
      uint32_t list_id = centroid_dists[pi].second;
      for (const auto& entry : data.lists[list_id]) {
        Candidate cand;
        cand.doc_id = entry.doc_id;
        float term_xy = 0.0f;

        if (use_rvq) {
          // 根据码本计算近似向量内积
          float approx_dot = 0.0f;
          for (size_t l = 0; l < entry.codes.size(); ++l) {
            approx_dot += lut(l, entry.codes[l]);
          }
          term_xy = approx_dot;
        } else {
          // 直接float向量相乘
          term_xy = entry.vector.dot(qw);
        }
            
        // |x - y|^2 = |x|^2 + |y|^2 - 2 * <x, y>
        cand.approx_dist = qnorm + entry.norm - 2.0f * term_xy;

        cand.rerank_dist = cand.approx_dist;
        cand.versions = entry.versions;
        cand.versions.index_version = data.version;
        cand.from_new = from_new;
        candidates.push_back(std::move(cand));
      }
    }

    if (candidates.empty()) {
      return candidates;
    }

    if (candidates.size() > topk) {
      std::nth_element(candidates.begin(), candidates.begin() + topk, candidates.end(),
                       [](const Candidate& a, const Candidate& b) { return a.approx_dist < b.approx_dist; });
      candidates.resize(topk);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.approx_dist < b.approx_dist; });
    return candidates;
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

  void SetQuantizer(std::shared_ptr<RVQCodebook> quantizer, VersionId ver) override {
    quantizer_ = quantizer;
    quantizer_ver_ = ver;
  }
 private:
  std::shared_ptr<RVQCodebook> quantizer_;
  VersionId quantizer_ver_{0};
  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, std::unique_ptr<IndexData>> data_map_;
  VersionId next_version_{1};
  VersionId latest_version_{0};
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<KMeansIVFIndex>(); }

}  // namespace ann
