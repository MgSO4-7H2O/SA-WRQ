#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include <Eigen/Dense>

#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/IndexIVF.h>

#include "common/config.h"
#include "common/dataset.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

  std::string BuildFaissFactoryKey(const Config& config, bool use_rvq) {
    if (!use_rvq) {
      return "IVF" + std::to_string(config.ivf_nlist) + ",Flat";
  }
  const uint32_t nbits = static_cast<uint32_t>(std::log2(config.rvq_codewords));
  return "IVF" + std::to_string(config.ivf_nlist) + ",RQ" +
         std::to_string(config.rvq_layers) + "x" + std::to_string(nbits);  
}

}  // namespace

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> base_dataset_path;
  if (argc > 2) {
    auto resolved = ResolveFvecsPath(argv[2], "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
  }

  if (!base_dataset_path) {
    std::cerr << "[ERROR] build_index requires a base dataset path." << std::endl;
    return 1;
  }

  auto matrix_res = LoadFvecs(*base_dataset_path);
  if (!matrix_res.ok()) {
    std::cerr << matrix_res.status().ToString() << std::endl;
    return 1;
  }
  MatrixRM X = matrix_res.value();
  if (config.dim != static_cast<uint32_t>(X.cols())) {
    std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << X.cols() << std::endl;
    config.dim = static_cast<uint32_t>(X.cols());
  }

  MatrixRM X_train = X;
  auto whitening = CreateWhiteningModel();
  if (config.use_whitening) {
    auto whiten_version_res = whitening->Fit(X);
    if (!whiten_version_res.ok()) {
      std::cerr << whiten_version_res.status().ToString() << std::endl;
      return 1;
    }
    auto batch_res = whitening->TransformBatch(X, whiten_version_res.value());
    if (!batch_res.ok()) {
      std::cerr << batch_res.status().ToString() << std::endl;
      return 1;
    }
    X_train = batch_res.value();
  }

  const bool use_rvq = true;
  const std::string key = BuildFaissFactoryKey(config, use_rvq);
  std::unique_ptr<faiss::Index> index(
      faiss::index_factory(static_cast<int>(config.dim), key.c_str(), faiss::METRIC_L2));
  if (!index) {
    std::cerr << "[ERROR] Failed to create faiss index: " << key << std::endl;
    return 1;
  }

  index->train(X_train.rows(), X_train.data());
  index->add(X_train.rows(), X_train.data());

  const std::string out_path = (argc > 3) ? argv[3] : "faiss.index";
  faiss::write_index(index.get(), out_path.c_str());
  std::cout << "[INFO] Wrote faiss index to " << out_path << std::endl;
  return 0;
}
