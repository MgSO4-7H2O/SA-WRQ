#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "quant/rvq.h"
#include "search/exact_search.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

// --- 辅助：计算 RVQ 重构误差 (MSE) ---
// 用于检查 RVQ 是否在当前数据分布下有效收敛
double ComputeRVQMSE(std::shared_ptr<RVQCodebook> rvq, VersionId ver, 
                     const MatrixRM& Data, size_t num_samples = 1000) {
    double total_err = 0;
    size_t samples = std::min((size_t)Data.rows(), num_samples);
    std::vector<uint32_t> codes;
    Eigen::VectorXf residual;
    
    for(size_t i=0; i<samples; ++i) {
        Eigen::VectorXf vec = Data.row(i).transpose();
        rvq->Encode(vec, ver, &codes, &residual);
        total_err += residual.squaredNorm();
    }
    return total_err / samples;
}

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

  std::vector<DocId> ids(X.rows());
  std::iota(ids.begin(), ids.end(), 0);

  // --- 4. 主循环测试 ---
  for (const auto& mode : modes) {
      std::cout << "\n==========================================" << std::endl;
      std::cout << "Running Mode: " << mode.name << std::endl;
      std::cout << "==========================================" << std::endl;

      // 每次重新创建组件，确保隔离
      auto ivf = CreateIVFIndex();
      auto rvq = CreateRVQCodebook();
      auto whitening = CreateWhiteningModel();
      
      auto searcher_res = CreateHybridSearcher(config);
      if (!searcher_res.ok()) return 1;
      auto searcher = std::move(searcher_res.value());

      MatrixRM X_train = X; // X_train 随着处理流程可能变化 (原始 -> 白化)
      VersionId v_whiten = 0;
      VersionId v_rvq = 0;
      VersionId v_ivf = 0;

      // --- Step 1: 白化训练与转换 ---
      if (mode.use_whiten) {
          v_whiten = whitening->Fit(X).value();
          // 将 Base 数据转换到白化空间
          X_train = whitening->TransformBatch(X, v_whiten).value();
          std::cout << "[Step] Whitening trained (ver " << v_whiten << ")" << std::endl;
      }

      // --- Step 2: RVQ 训练 ---
      double rvq_mse = 0.0;
      if (mode.use_rvq) {
          RVQParams p; 
          p.num_layers = config.rvq_layers;
          p.codewords = config.rvq_codewords;
          
          // 对 X_train 进行训练 (如果开了白化，这里就是白化后的数据)
          v_rvq = rvq->Train(X_train, p).value();
          
          rvq_mse = ComputeRVQMSE(rvq, v_rvq, X_train);
          std::cout << "[Step] RVQ trained (ver " << v_rvq << "), MSE=" << rvq_mse << std::endl;
          ivf->SetQuantizer(rvq, v_rvq);
      } else {
        // 不使用 RVQ，只用传空量化器
          ivf->SetQuantizer(nullptr, 0);
      }

      // --- Step 3: IVF 构建 ---
      IVFParams ivf_p;
      ivf_p.nlist = config.ivf_nlist;
      ivf_p.dim = config.dim;
      // 使用 X_train 构建索引
      // RVQ存储Codes, Plain存储原始向量
      v_ivf = ivf->Build(X_train, ids, ivf_p, 0).value();
      std::cout << "[Step] IVF built" << std::endl;

      // --- Step 4: 组装 Searcher ---
      VersionSet vers{v_whiten, v_rvq, v_ivf};
      searcher->SetIndex(ivf, vers);
      
      if (mode.use_whiten) {
          // 启用白化：Searcher 会自动将 Query 变换到白化空间
          searcher->SetWhitening(whitening, v_whiten);
      } else {
          searcher->SetWhitening(nullptr, 0);
      }

      // --- Step 5: 执行搜索 ---
      SearchParams sp;
      sp.topk = config.topk;
      sp.nprobe = config.nprobe;
      sp.use_whitening = mode.use_whiten;

      std::vector<std::vector<DocId>> preds(Q.rows());
      Timer t_total;

      for (int i = 0; i < Q.rows(); ++i) {
          Eigen::VectorXf q = Q.row(i).transpose();
          // 始终传入原始 Query，由 Searcher 内部根据配置决定是否 Transform
          auto res = searcher->Search(q, sp);
          
          if(res.ok()) {
              for(auto& c : res.value().topk) preds[i].push_back(c.doc_id);
          }
      }

      // --- Step 6: 统计结果 ---
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