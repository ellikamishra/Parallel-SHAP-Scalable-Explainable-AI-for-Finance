\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <omp.h>

namespace py = pybind11;

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

PYBIND11_MODULE(mc_shap_openmp, m) {
    m.doc() = "Monte-Carlo SHAP with OpenMP (model-agnostic, calls back into Python model)";
    m.def("mc_shap_openmp", &mc_shap_openmp,
          py::arg("f"), py::arg("X"), py::arg("X_bg"), py::arg("P")=128, py::arg("seed_base")=0u);
}
