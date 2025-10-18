\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <curand_kernel.h>
#include <cmath>

namespace py = pybind11;
__device__ inline double sigmoid(double z){ return 1.0 / (1.0 + exp(-z)); }

__global__ void mc_shap_linear_kernel(
    const double* __restrict__ X,
    const double* __restrict__ baseline,
    const double* __restrict__ W,
    const double  b,
    double* __restrict__ out,
    int N, int D, int P, unsigned long long seed_base
){
    int i = blockIdx.x;
    if (i >= N) return;

    extern __shared__ double sh[];
    double* a = sh;
    for (int j = threadIdx.x; j < D; j += blockDim.x) a[j] = baseline[j];
    __syncthreads();

    int P_thread = (P + blockDim.x - 1) / blockDim.x;
    int p_start = threadIdx.x * P_thread;
    int p_end = min(P, p_start + P_thread);

    for (int p = p_start; p < p_end; ++p) {
        __syncthreads();
        for (int j = threadIdx.x; j < D; j += blockDim.x) a[j] = baseline[j];
        __syncthreads();

        double z0 = 0.0;
        for (int j = threadIdx.x; j < D; j += blockDim.x) z0 += W[j] * a[j];
        __shared__ double red;
        if (threadIdx.x == 0) red = 0.0;
        __syncthreads();
        atomicAdd(&red, z0);
        __syncthreads();
        double prev = 1.0 / (1.0 + exp(-(red + b)));
        __syncthreads();

        for (int k = 0; k < D; ++k) {
            int feat = (k * 1315423911u + p * 2654435761u) % D;
            if (threadIdx.x == 0) a[feat] = X[i*D + feat];
            __syncthreads();

            double z = 0.0;
            for (int j = threadIdx.x; j < D; j += blockDim.x) z += W[j] * a[j];
            __syncthreads();
            if (threadIdx.x == 0) red = 0.0;
            __syncthreads();
            atomicAdd(&red, z);
            __syncthreads();
            double cur = 1.0 / (1.0 + exp(-(red + b)));
            __syncthreads();

            if (threadIdx.x == 0) atomicAdd(&out[i*D + feat], (cur - prev) / double(P));
            prev = cur;
            __syncthreads();
        }
    }
}

py::array_t<double> mc_shap_cuda_linear(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> baseline,
    py::array_t<double, py::array::c_style | py::array::forcecast> W,
    double b,
    int P,
    int threads_per_block,
    unsigned long long seed_base
){
    auto bX = X.request(); auto bB = baseline.request(); auto bW = W.request();
    int N = bX.shape[0], D = bX.shape[1];

    double *dX, *dB, *dW, *dOut;
    cudaMalloc(&dX, sizeof(double)*N*D);
    cudaMalloc(&dB, sizeof(double)*D);
    cudaMalloc(&dW, sizeof(double)*D);
    cudaMalloc(&dOut, sizeof(double)*N*D);
    cudaMemset(dOut, 0, sizeof(double)*N*D);

    cudaMemcpy(dX, bX.ptr, sizeof(double)*N*D, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, bB.ptr, sizeof(double)*D,    cudaMemcpyHostToDevice);
    cudaMemcpy(dW, bW.ptr, sizeof(double)*D,    cudaMemcpyHostToDevice);

    int blocks = N;
    size_t shmem = sizeof(double)*D;
    mc_shap_linear_kernel<<<blocks, threads_per_block, shmem>>>(
        dX, dB, (double*)bW.ptr, b, dOut, N, D, P, seed_base
    );
    cudaDeviceSynchronize();

    auto out = py::array_t<double>({N, D});
    cudaMemcpy(out.request().ptr, dOut, sizeof(double)*N*D, cudaMemcpyDeviceToHost);

    cudaFree(dX); cudaFree(dB); cudaFree(dW); cudaFree(dOut);
    return out;
}

PYBIND11_MODULE(mc_shap_cuda, m){
    m.doc() = "CUDA Monte-Carlo SHAP for linear model (demo)";
    m.def("mc_shap_cuda_linear", &mc_shap_cuda_linear,
          py::arg("X"), py::arg("baseline"), py::arg("W"), py::arg("b"),
          py::arg("P")=128, py::arg("threads_per_block")=128, py::arg("seed_base")=0ull);
}
