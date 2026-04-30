#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <string>

namespace py = pybind11;

// Hash function for vectors (for caching predictions)
struct VectorHash {
    std::size_t operator()(const std::vector<double>& v) const {
        std::size_t seed = v.size();
        for (auto& i : v) {
            seed ^= std::hash<double>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Original version (slow but kept for compatibility)
py::array_t<double> mc_shap_openmp(
    py::object f,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> X_bg,
    int P,
    unsigned int seed_base
) {
    auto bufX = X.request();
    auto bufB = X_bg.request();

    const int64_t N = bufX.shape[0];
    const int64_t D = bufX.shape[1];
    const int64_t M = bufB.shape[0];

    const double* Xp = static_cast<double*>(bufX.ptr);
    const double* Bp = static_cast<double*>(bufB.ptr);

    std::vector<double> baseline(D, 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const double* row = Bp + m*D;
        for (int64_t j = 0; j < D; ++j) baseline[j] += row[j];
    }
    for (int64_t j = 0; j < D; ++j) baseline[j] /= double(M);

    auto out = py::array_t<double>({N, D});
    auto bufOut = out.request();
    double* Op = static_cast<double*>(bufOut.ptr);
    std::fill(Op, Op + N*D, 0.0);

    const py::array::ShapeContainer shape{py::ssize_t(1), py::ssize_t(D)};

    #pragma omp parallel
    {
        std::vector<int> perm(D);
        std::vector<double> a(D, 0.0);
        std::vector<double> phi(D, 0.0);

        auto call_model = [&](const std::vector<double>& data) -> double {
            py::gil_scoped_acquire gil;
            py::array_t<double> arr(shape);
            auto info = arr.request();
            double* ptr = static_cast<double*>(info.ptr);
            std::copy(data.begin(), data.end(), ptr);
            py::object out_obj = f(arr);
            if (py::isinstance<py::float_>(out_obj) || py::isinstance<py::int_>(out_obj)) {
                return out_obj.cast<double>();
            }
            py::array out_arr = py::array::ensure(out_obj);
            if (!out_arr) {
                throw std::runtime_error("Model function must return a numeric scalar or numpy array.");
            }
            auto out_info = out_arr.request();
            if (out_info.size <= 0) {
                throw std::runtime_error("Model function must return an array of at least one value.");
            }
            return static_cast<double*>(out_info.ptr)[0];
        };

        std::mt19937_64 rng(seed_base + omp_get_thread_num());

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            const double* xi = Xp + i*D;
            std::fill(phi.begin(), phi.end(), 0.0);

            for (int p = 0; p < P; ++p) {
                std::iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), rng);

                for (int64_t j = 0; j < D; ++j) {
                    a[j] = baseline[j];
                }

                double prev = call_model(a);

                for (int k = 0; k < D; ++k) {
                    int feat = perm[k];
                    a[feat] = xi[feat];
                    double cur = call_model(a);
                    phi[feat] += (cur - prev);
                    prev = cur;
                }
            }

            double* rowO = Op + i*D;
            for (int64_t j = 0; j < D; ++j) {
                rowO[j] = phi[j] / double(P);
            }
        }
    }

    return out;
}

// OPTIMIZED VERSION: Pre-compute all predictions first
py::array_t<double> mc_shap_openmp_fast(
    py::object f,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> X_bg,
    int P,
    unsigned int seed_base
) {
    auto bufX = X.request();
    auto bufB = X_bg.request();

    const int64_t N = bufX.shape[0];
    const int64_t D = bufX.shape[1];
    const int64_t M = bufB.shape[0];

    const double* Xp = static_cast<double*>(bufX.ptr);
    const double* Bp = static_cast<double*>(bufB.ptr);

    // Compute baseline
    std::vector<double> baseline(D, 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const double* row = Bp + m*D;
        for (int64_t j = 0; j < D; ++j) baseline[j] += row[j];
    }
    for (int64_t j = 0; j < D; ++j) baseline[j] /= double(M);

    // STEP 1 (parallel): each thread walks its share of samples with the SAME
    // per-thread RNG stream that STEP 3 will later use, and fills a thread-local
    // dedup set. After the parallel region, the local sets are merged serially.
    // Matching the RNG seeding (seed_base + tid) and schedule(static) between
    // STEP 1 and STEP 3 guarantees every vector STEP 3 looks up was cached here.
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<std::vector<double>>> local_samples(nthreads);
    std::vector<std::unordered_map<std::vector<double>, size_t, VectorHash>> local_idx(nthreads);

    {
        py::gil_scoped_release release_gil;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& lm = local_idx[tid];
            auto& ls = local_samples[tid];
            std::mt19937_64 rng(seed_base + tid);
            std::vector<int> perm(D);
            std::vector<double> a(D);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < N; ++i) {
                const double* xi = Xp + i*D;

                for (int p = 0; p < P; ++p) {
                    std::iota(perm.begin(), perm.end(), 0);
                    std::shuffle(perm.begin(), perm.end(), rng);

                    std::copy(baseline.begin(), baseline.end(), a.begin());
                    if (lm.find(a) == lm.end()) {
                        lm[a] = ls.size();
                        ls.push_back(a);
                    }

                    for (int k = 0; k < D; ++k) {
                        int feat = perm[k];
                        a[feat] = xi[feat];
                        if (lm.find(a) == lm.end()) {
                            lm[a] = ls.size();
                            ls.push_back(a);
                        }
                    }
                }
            }
        }
    }

    // Serial merge of thread-local caches into the global cache.
    std::vector<std::vector<double>> all_samples;
    std::unordered_map<std::vector<double>, size_t, VectorHash> sample_to_idx;
    {
        size_t total = 0;
        for (auto& ls : local_samples) total += ls.size();
        all_samples.reserve(total);
        sample_to_idx.reserve(total);
        for (auto& ls : local_samples) {
            for (auto& v : ls) {
                if (sample_to_idx.find(v) == sample_to_idx.end()) {
                    sample_to_idx[v] = all_samples.size();
                    all_samples.push_back(std::move(v));
                }
            }
            ls.clear(); ls.shrink_to_fit();
        }
        local_idx.clear(); local_idx.shrink_to_fit();
        local_samples.clear(); local_samples.shrink_to_fit();
    }
    
    // STEP 2: Batch predict all samples in Python (ONE GIL acquire)
    std::vector<double> predictions(all_samples.size());
    {
        py::gil_scoped_acquire gil;
        
        // Create batch array
        py::array_t<double> batch_arr({(py::ssize_t)all_samples.size(), (py::ssize_t)D});
        auto batch_info = batch_arr.request();
        double* batch_ptr = static_cast<double*>(batch_info.ptr);
        
        // Fill batch
        for (size_t i = 0; i < all_samples.size(); ++i) {
            std::copy(all_samples[i].begin(), all_samples[i].end(), 
                     batch_ptr + i * D);
        }
        
        // Call model ONCE with entire batch
        py::object out_obj = f(batch_arr);
        py::array out_arr = py::array::ensure(out_obj);
        if (!out_arr) {
            throw std::runtime_error("Model must return array");
        }
        
        auto out_info = out_arr.request();
        double* out_ptr = static_cast<double*>(out_info.ptr);
        
        // Store predictions
        for (size_t i = 0; i < all_samples.size(); ++i) {
            predictions[i] = out_ptr[i];
        }
    }
    
    // STEP 3: Compute SHAP values using cached predictions (no GIL!)
    auto out = py::array_t<double>({N, D});
    auto bufOut = out.request();
    double* Op = static_cast<double*>(bufOut.ptr);
    std::fill(Op, Op + N*D, 0.0);
    
    {
        py::gil_scoped_release release_gil;
        
        #pragma omp parallel
        {
            std::vector<int> perm(D);
            std::vector<double> a(D);
            std::vector<double> phi(D, 0.0);
            std::mt19937_64 rng(seed_base + omp_get_thread_num());
            
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < N; ++i) {
                const double* xi = Xp + i*D;
                std::fill(phi.begin(), phi.end(), 0.0);
                
                for (int p = 0; p < P; ++p) {
                    std::iota(perm.begin(), perm.end(), 0);
                    std::shuffle(perm.begin(), perm.end(), rng);
                    
                    std::copy(baseline.begin(), baseline.end(), a.begin());
                    double prev = predictions[sample_to_idx.at(a)];
                    
                    for (int k = 0; k < D; ++k) {
                        int feat = perm[k];
                        a[feat] = xi[feat];
                        double cur = predictions[sample_to_idx.at(a)];
                        phi[feat] += (cur - prev);
                        prev = cur;
                    }
                }
                
                double* rowO = Op + i*D;
                for (int64_t j = 0; j < D; ++j) {
                    rowO[j] = phi[j] / double(P);
                }
            }
        }
    }
    
    return out;
}

PYBIND11_MODULE(mc_shap_openmp, m) {
    m.doc() = "Monte-Carlo SHAP with OpenMP";
    
    m.def("mc_shap_openmp", &mc_shap_openmp,
          py::arg("f"), py::arg("X"), py::arg("X_bg"), 
          py::arg("P")=128, py::arg("seed_base")=0u,
          "Original version (slow due to GIL contention)");
    
    m.def("mc_shap_openmp_fast", &mc_shap_openmp_fast,
          py::arg("f"), py::arg("X"), py::arg("X_bg"), 
          py::arg("P")=128, py::arg("seed_base")=0u,
          "Optimized version with batch predictions");
}