#include <algorithm>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/result.h"
#include "common/types.h"
#include "index/ivf.h"
#include "quant/rvq.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

MatrixRM GenerateRandomMatrix(uint32_t rows, uint32_t cols, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  MatrixRM m(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      m(r, c) = dist(gen);
    }
  }
  return m;
}

void LogStatus(const Status& status, const std::string& step) {
  if (!status.ok()) {
    std::cerr << "[ERROR] " << step << ": " << status.ToString() << std::endl;
  } else {
    std::cout << "[INFO] " << step << " OK" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  const Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  auto whitening = CreateWhiteningModel();
  auto rvq = CreateRVQCodebook();
  auto ivf = CreateIVFIndex();

  const uint32_t num_vectors = 32;
  MatrixRM X = GenerateRandomMatrix(num_vectors, config.dim, config.seed);

  auto whiten_version_res = whitening->Fit(X);
  if (!whiten_version_res.ok()) {
    std::cerr << whiten_version_res.status().ToString() << std::endl;
    return 1;
  }
  VersionId whiten_version = whiten_version_res.value();

  RVQParams rvq_params;
  rvq_params.num_layers = config.rvq_layers;
  rvq_params.codewords = config.rvq_codewords;

  auto rvq_version_res = rvq->Train(X, rvq_params);
  if (!rvq_version_res.ok()) {
    std::cerr << rvq_version_res.status().ToString() << std::endl;
    return 1;
  }
  VersionId rvq_version = rvq_version_res.value();

  std::vector<DocId> ids(num_vectors);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    ids[i] = i;
  }

  IVFParams ivf_params;
  ivf_params.nlist = std::max(1u, config.ivf_nlist);
  ivf_params.dim = config.dim;
  auto ivf_version_res = ivf->Build(X, ids, ivf_params, 1);
  if (!ivf_version_res.ok()) {
    std::cerr << ivf_version_res.status().ToString() << std::endl;
    return 1;
  }
  VersionId ivf_version = ivf_version_res.value();

  AlignedVector<VectorRecord> records;
  records.reserve(num_vectors);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    VectorRecord rec;
    rec.doc_id = ids[i];
    rec.dim = config.dim;
    rec.versions = VersionSet{whiten_version, rvq_version, ivf_version};
    rec.ivf_id = i % std::max(1u, ivf_params.nlist);
    rec.x = X.row(i).transpose();

    Eigen::VectorXf xw(config.dim);
    auto transform_res = whitening->Transform(rec.x, whiten_version, xw);
    if (!transform_res.ok()) {
      std::cerr << transform_res.status().ToString() << std::endl;
      return 1;
    }

    Eigen::VectorXf residual(config.dim);
    std::vector<uint32_t> codes;
    auto encode_res = rvq->Encode(xw, rvq_version, &codes, &residual);
    if (!encode_res.ok()) {
      std::cerr << encode_res.status().ToString() << std::endl;
      return 1;
    }
    rec.codes = std::move(codes);
    records.push_back(rec);
  }

  Status add_status = ivf->Add(records);
  LogStatus(add_status, "IVF::Add");

  auto bytes = whitening->Serialize();
  if (!bytes.ok()) {
    std::cerr << bytes.status().ToString() << std::endl;
  }
  std::cout << "Serialized whitening bytes: " << bytes.value().size() << std::endl;

  return add_status.ok() ? 0 : 1;
}
