#include "index/ivf.h"

#include <algorithm>
#include <cstdint>
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
  Eigen::VectorXf vector;
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

int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const IndexData& data) {
  float best = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < data.centroids.rows(); ++i) {
    float dist = (data.centroids.row(i).transpose() - vec).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

MatrixRM InitializeCentroids(Eigen::Ref<const MatrixRM> X, uint32_t nlist) {
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

  for (int iter = 0; iter < kKMeansIterations; ++iter) {
    // Assignment step.
    for (int64_t i = 0; i < num_vecs; ++i) {
      Eigen::VectorXf vec = X.row(i).transpose();
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

    // Update step.
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(k, 0);
    for (int64_t i = 0; i < num_vecs; ++i) {
      new_centroids.row(assignments[i]) += X.row(i);
      counts[assignments[i]]++;
    }
    std::mt19937 gen(kDefaultSeed + iter);
    std::uniform_int_distribution<int64_t> dist_index(0, num_vecs - 1);
    for (int64_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        new_centroids.row(c) /= static_cast<float>(counts[c]);
      } else {
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

    // Assign vectors to lists.
    for (int64_t i = 0; i < Xw.rows(); ++i) {
      Eigen::VectorXf vec = Xw.row(i).transpose();
      float best = std::numeric_limits<float>::max();
      int best_idx = 0;
      for (uint32_t c = 0; c < nlist; ++c) {
        float dist = (data->centroids.row(c).transpose() - vec).squaredNorm();
        if (dist < best) {
          best = dist;
          best_idx = static_cast<int>(c);
        }
      }
      ListEntry entry;
      entry.doc_id = ids[i];
      entry.versions = VersionSet{0, 0, 0};
      entry.vector = std::move(vec);
      entry.norm = entry.vector.squaredNorm();
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
      int centroid = NearestCentroid(rec.x, data);
      ListEntry entry;
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      entry.vector = rec.x;
      entry.norm = entry.vector.squaredNorm();
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

    std::vector<Candidate> candidates;
    const float qnorm = qw.squaredNorm();
    for (uint32_t pi = 0; pi < probes; ++pi) {
      uint32_t list_id = centroid_dists[pi].second;
      for (const auto& entry : data.lists[list_id]) {
        Candidate cand;
        cand.doc_id = entry.doc_id;
        const float dot = entry.vector.dot(qw);
        cand.approx_dist = qnorm + entry.norm - 2.0f * dot;
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

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, std::unique_ptr<IndexData>> data_map_;
  VersionId next_version_{1};
  VersionId latest_version_{0};
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<KMeansIVFIndex>(); }

}  // namespace ann
