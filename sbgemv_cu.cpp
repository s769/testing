#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>
#include <complex>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

// Error checking macros
#define CUDA_CHECK(err)                                                                                                    \
    {                                                                                                                      \
        cudaError_t err_ = (err);                                                                                          \
        if (err_ != cudaSuccess)                                                                                           \
        {                                                                                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                            \
        }                                                                                                                  \
    }

#define CUBLAS_CHECK(err)                                                                                                       \
    {                                                                                                                           \
        cublasStatus_t err_ = (err);                                                                                            \
        if (err_ != CUBLAS_STATUS_SUCCESS)                                                                                      \
        {                                                                                                                       \
            std::cerr << "cuBLAS error: " << cublasGetStatusString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }

// Overload operators for cuComplex types for the CPU reference implementation
static __inline__ cuComplex operator*(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
static __inline__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
static __inline__ cuComplex operator+(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
static __inline__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
static __inline__ cuComplex operator-(cuComplex a, cuComplex b) { return cuCsubf(a, b); }
static __inline__ cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }
static __inline__ float abs(cuComplex a) { return cuCabsf(a); }
static __inline__ double abs(cuDoubleComplex a) { return cuCabs(a); }
static __inline__ cuComplex conj(cuComplex a) { return cuConjf(a); }
static __inline__ cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }

// Helper to get cuBLAS function
template <typename T>
struct CublasGemvStridedBatched;

template <>
struct CublasGemvStridedBatched<float>
{
    static constexpr auto value = cublasSgemvStridedBatched;
};
template <>
struct CublasGemvStridedBatched<double>
{
    static constexpr auto value = cublasDgemvStridedBatched;
};
template <>
struct CublasGemvStridedBatched<cuComplex>
{
    static constexpr auto value = cublasCgemvStridedBatched;
};
template <>
struct CublasGemvStridedBatched<cuDoubleComplex>
{
    static constexpr auto value = cublasZgemvStridedBatched;
};

// CPU verification function
template <typename T>
void cpu_gemv_strided_batched(cublasOperation_t trans, int m, int n, const T &alpha, const std::vector<T> &A, const std::vector<T> &x, const T &beta, std::vector<T> &y, int batch_count)
{
    int y_dim = (trans == CUBLAS_OP_N) ? m : n;
    int dot_dim = (trans == CUBLAS_OP_N) ? n : m;

    for (int b = 0; b < batch_count; ++b)
    {
        const T *current_A = A.data() + static_cast<size_t>(b) * m * n;
        const T *current_x = x.data() + static_cast<size_t>(b) * dot_dim;
        T *current_y = y.data() + static_cast<size_t>(b) * y_dim;

        for (int i = 0; i < y_dim; ++i)
        {
            T sum{};
            if constexpr (std::is_same_v<T, cuComplex>)
            {
                sum = make_cuComplex(0.0, 0.0);
            }
            else if constexpr (std::is_same_v<T, cuDoubleComplex>)
            {
                sum = make_cuDoubleComplex(0.0, 0.0);
            }
            else
            {
                sum = 0.0;
            }

            for (int j = 0; j < dot_dim; ++j)
            {
                T a_val;
                if (trans == CUBLAS_OP_N)
                {
                    a_val = current_A[i + static_cast<size_t>(j) * m]; // Access A(i,j) from column-major A
                }
                else
                {
                    a_val = current_A[j + static_cast<size_t>(i) * m]; // Access A(j,i) for transpose
                }

                if constexpr (std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>)
                {
                    if (trans == CUBLAS_OP_C)
                    {
                        a_val = conj(a_val);
                    }
                }
                T x_val = current_x[j];
                sum = sum + a_val * x_val;
            }
            current_y[i] = alpha * sum;
        }
    }
}

template <typename T>
void run_benchmark(const std::string &trans_str, int m, int n, int batch_count, bool verify)
{
    cublasOperation_t trans;
    if (trans_str == "N" || trans_str == "n")
        trans = CUBLAS_OP_N;
    else if (trans_str == "T" || trans_str == "t")
        trans = CUBLAS_OP_T;
    else if (trans_str == "H" || trans_str == "h" || trans_str == "C" || trans_str == "c")
        trans = CUBLAS_OP_C; // cuBLAS uses 'C' for conjugate
    else
    {
        std::cerr << "Invalid transpose operation specified. Use N, T, or H/C." << std::endl;
        exit(EXIT_FAILURE);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Determine vector sizes based on operation
    int x_dim = (trans == CUBLAS_OP_N) ? n : m;
    int y_dim = (trans == CUBLAS_OP_N) ? m : n;

    // Host data
    std::vector<T> h_A(static_cast<size_t>(m) * n * batch_count);
    std::vector<T> h_x(static_cast<size_t>(x_dim) * batch_count);
    std::vector<T> h_y(static_cast<size_t>(y_dim) * batch_count);

    // Initialize data with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 2.0);

    for (auto &val : h_A)
    {
        if constexpr (std::is_same_v<T, cuComplex>)
        {
            val = make_cuComplex(dis(gen), dis(gen));
        }
        else if constexpr (std::is_same_v<T, cuDoubleComplex>)
        {
            val = make_cuDoubleComplex(dis(gen), dis(gen));
        }
        else
        {
            val = dis(gen);
        }
    }
    for (auto &val : h_x)
    {
        if constexpr (std::is_same_v<T, cuComplex>)
        {
            val = make_cuComplex(dis(gen), dis(gen));
        }
        else if constexpr (std::is_same_v<T, cuDoubleComplex>)
        {
            val = make_cuDoubleComplex(dis(gen), dis(gen));
        }
        else
        {
            val = dis(gen);
        }
    }

    // Device data
    T *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_y, h_y.size() * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(T), cudaMemcpyHostToDevice));

    // Strides
    long long int stride_A = static_cast<long long int>(m) * n;
    long long int stride_x = x_dim;
    long long int stride_y = y_dim;

    T alpha, beta;
    if constexpr (std::is_same_v<T, cuComplex>)
    {
        alpha = make_cuComplex(1.0, 0.0);
        beta = make_cuComplex(0.0, 0.0);
    }
    else if constexpr (std::is_same_v<T, cuDoubleComplex>)
    {
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta = make_cuDoubleComplex(0.0, 0.0);
    }
    else
    {
        alpha = 1.0;
        beta = 0.0;
    }

    // Warm-up run
    CUBLAS_CHECK(CublasGemvStridedBatched<T>::value(handle, trans, m, n, &alpha, d_A, m, stride_A, d_x, 1, stride_x, &beta, d_y, 1, stride_y, batch_count));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    const int num_runs = 100;
    for (int i = 0; i < num_runs; ++i)
    {
        CUBLAS_CHECK(CublasGemvStridedBatched<T>::value(handle, trans, m, n, &alpha, d_A, m, stride_A, d_x, 1, stride_x, &beta, d_y, 1, stride_y, batch_count));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "trans='" << trans_str << "', m=" << m << ", n=" << n << ", batch_count=" << batch_count << std::endl;
    std::cout << "Average execution time: " << milliseconds / num_runs << " ms" << std::endl;

    if (verify)
    {
        std::cout << "\nVerifying result..." << std::endl;
        std::vector<T> h_y_from_gpu(h_y.size());
        CUDA_CHECK(cudaMemcpy(h_y_from_gpu.data(), d_y, h_y_from_gpu.size() * sizeof(T), cudaMemcpyDeviceToHost));

        std::vector<T> h_y_cpu(h_y.size());
        cpu_gemv_strided_batched(trans, m, n, alpha, h_A, h_x, beta, h_y_cpu, batch_count);

        double max_error = 0.0;
        for (size_t i = 0; i < h_y_cpu.size(); ++i)
        {
            double error = abs(h_y_cpu[i] - h_y_from_gpu[i]);
            if (error > max_error)
            {
                max_error = error;
            }
        }

        double tolerance = 1e-5;
        std::cout << "Max error: " << max_error << std::endl;
        if (max_error > tolerance)
        {
            std::cout << "Verification FAILED" << std::endl;
        }
        else
        {
            std::cout << "Verification PASSED" << std::endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 7)
    {
        std::cerr << "Usage: " << argv[0] << " <trans> <m> <n> <batch_count> <datatype> [verify]" << std::endl;
        std::cerr << "  trans: N (none), T (transpose), H or C (conjugate transpose)" << std::endl;
        std::cerr << "  datatype: s (float), d (double), c (complex), z (double complex)" << std::endl;
        return 1;
    }

    std::string trans_str = argv[1];
    int m = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    int batch_count = std::stoi(argv[4]);
    std::string datatype = argv[5];
    bool verify = false;
    if (argc == 7 && std::string(argv[6]) == "verify")
    {
        verify = true;
    }

    if (datatype == "s")
    {
        run_benchmark<float>(trans_str, m, n, batch_count, verify);
    }
    else if (datatype == "d")
    {
        run_benchmark<double>(trans_str, m, n, batch_count, verify);
    }
    else if (datatype == "c")
    {
        run_benchmark<cuComplex>(trans_str, m, n, batch_count, verify);
    }
    else if (datatype == "z")
    {
        run_benchmark<cuDoubleComplex>(trans_str, m, n, batch_count, verify);
    }
    else
    {
        std::cerr << "Invalid datatype. Use 's', 'd', 'c', or 'z'." << std::endl;
        return 1;
    }

    return 0;
}
