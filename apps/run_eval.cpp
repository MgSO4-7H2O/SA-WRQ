#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <faiss/index_factory.h>
#include <faiss/IndexIVF.h>

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "search/exact_search.h"
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
    
    double ComputeFaissMSE(faiss::Index* index, const MatrixRM& data, size_t num_samples = 1000) {
      const size_t samples = std::min(num_samples, static_cast<size_t>(data.rows()));
      if (samples == 0) {
        return 0.0;
      }
      double total_err = 0.0;
      std::vector<float> reconstructed(data.cols());
      std::vector<uint8_t> codes(index->sa_code_size());
      for (size_t i = 0; i < samples; ++i) {
        const float* vec = data.row(static_cast<int>(i)).data();
        index->sa_encode(1, vec, codes.data());
        index->sa_decode(1, codes.data(), reconstructed.data());
        Eigen::Map<const Eigen::VectorXf> rec(reconstructed.data(), data.cols());
        Eigen::VectorXf original = data.row(static_cast<int>(i)).transpose();
        total_err += (original - rec).squaredNorm();
      }
      return total_err / static_cast<double>(samples);
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
    std::cout << "Loaded Config: " << config_path << std::endl;

    // --- 1. load dataset ---
    std::string dataset_spec = (argc > 2) ? argv[2] : "";
    std::string query_spec = (argc > 3) ? argv[3] : "";

    MatrixRM X, Q;

    // 加载 Base 数据
    std::string base_path_resolved;
    if (!dataset_spec.empty()) {
        auto res = ResolveFvecsPath(dataset_spec, "_base.fvecs");
        if (res.ok()) base_path_resolved = res.value();
        else std::cout << "[Warn] Resolve base failed: " << res.status().ToString() << std::endl;
    }

    if (!base_path_resolved.empty()) {
        auto res = LoadFvecs(base_path_resolved);
        if (res.ok()) {
            X = res.value();
            std::cout << "[Data] Base loaded: " << X.rows() << "x" << X.cols() << " from " << base_path_resolved << std::endl;
        } else {
            std::cerr << "[Error] Failed to load base: " << res.status().ToString() << std::endl;
            return 1;
        }
    } else {
        std::cout << "[Warn] No dataset provided. Using Random(1000) Base Data" << std::endl;
        X = MatrixRM::Random(1000, config.dim);
    }

    // 加载 Query 数据
    std::string query_path_resolved;
    if (!query_spec.empty()) {
        auto res = ResolveFvecsPath(query_spec, "_query.fvecs");
        if (res.ok()) query_path_resolved = res.value();
    } 
    if (query_path_resolved.empty() && !dataset_spec.empty()) {
        auto res = ResolveFvecsPath(dataset_spec, "_query.fvecs");
        if (res.ok()) query_path_resolved = res.value();
    }

    if (!query_path_resolved.empty()) {
        auto res = LoadFvecs(query_path_resolved);
        if (res.ok()) {
            Q = res.value();
            if (Q.rows() > 100) Q = Q.topRows(100); // 内存保护
            std::cout << "[Data] Query loaded: " << Q.rows() << "x" << Q.cols() << std::endl;
        } else {
            std::cerr << "[Error] Failed to load query: " << res.status().ToString() << std::endl;
            return 1;
        }
    } else {
        std::cout << "[Warn] Using Random(10) Query Data" << std::endl;
        Q = MatrixRM::Random(10, config.dim);
    }

    // --- 2. 计算 Ground Truth (基于原始 L2 距离) ---
    std::cout << "[GT] Calculating Exact Ground Truth..." << std::endl;
    auto gt_res = ExactSearchBatch(Q, X, config.topk);
    if (!gt_res.ok()) return 1;
    auto ground_truth = gt_res.value();

    // --- 3. 定义四种测试模式 ---
    struct TestMode {
        std::string name;
        bool use_whiten;
        bool use_rvq;
    };

    std::vector<TestMode> modes = {
        {"1. IVF (Plain)",         false, false},
        {"2. IVF + Whiten",        true,  false},
        {"3. IVF + RVQ",           false, true},
        {"4. IVF + Whiten + RVQ",  true,  true}
    };

    // --- 4. 主循环测试 ---
    for (const auto& mode : modes) {
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Running Mode: " << mode.name << std::endl;
        std::cout << "==========================================" << std::endl;
    
        auto whitening = CreateWhiteningModel();
        MatrixRM X_train = X;
        VersionId v_whiten = 0;
    
        if (mode.use_whiten) {
            v_whiten = whitening->Fit(X).value();
            X_train = whitening->TransformBatch(X, v_whiten).value();
            std::cout << "[Step] Whitening trained (ver " << v_whiten << ")" << std::endl;
        }

        const std::string key = BuildFaissFactoryKey(config, mode.use_rvq);
        std::unique_ptr<faiss::Index> index( 
            faiss::index_factory(static_cast<int>(X_train.cols()), key.c_str(), faiss::METRIC_L2));
        if (!index) {
            std::cerr << "[Error] Failed to create faiss index: " << key << std::endl;
            return 1;
        }
    
        index->train(X_train.rows(), X_train.data());
        index->add(X_train.rows(), X_train.data());
    
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get())) {
            ivf->nprobe = static_cast<int>(config.nprobe);
        }
    
        double rvq_mse = 0.0;
        if (mode.use_rvq) {
            rvq_mse = ComputeFaissMSE(index.get(), X_train);
            std::cout << "[Step] RVQ trained (faiss), MSE=" << rvq_mse << std::endl;
        }
    
        std::vector<std::vector<DocId>> preds(Q.rows());
        Timer t_total;
        std::vector<float> distances(config.topk);
        std::vector<faiss::idx_t> labels(config.topk);
    
        for (int i = 0; i < Q.rows(); ++i) {
            Eigen::VectorXf q = Q.row(i).transpose();
            if (mode.use_whiten) {
                Eigen::VectorXf q_w(config.dim);
                auto status = whitening->Transform(q, v_whiten, q_w);
                if (!status.ok()) return 1;
                q = q_w;
            }
            index->search(1, q.data(), config.topk, distances.data(), labels.data());
            for (int k = 0; k < config.topk; ++k) {
                if (labels[k] >= 0) {
                    preds[i].push_back(static_cast<DocId>(labels[k]));
                }
            }
        }

        // 统计结果
        double recall = ann::RecallAtK(ground_truth, preds, config.topk).value();
        double qps = Q.rows() / (t_total.ElapsedMillis() / 1000.0);

        std::cout << ">>> RESULT: " << mode.name << std::endl;
        std::cout << "    Recall@" << config.topk << ": " << std::fixed << std::setprecision(4) << recall << std::endl;
        std::cout << "    QPS:       " << qps << std::endl;
        if (mode.use_rvq) {
            std::cout << "    RVQ MSE:   " << rvq_mse << std::endl;
        }
    }

    return 0;
}