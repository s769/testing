#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>
#include <complex> // Required for std::real, std::imag

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Error checking macros
#define HIP_CHECK(err)                                                                                                   \
    {                                                                                                                    \
        hipError_t err_ = (err);                                                                                         \
        if (err_ != hipSuccess)                                                                                          \
        {                                                                                                                \
            std::cerr << "HIP error: " << hipGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                          \
        }                                                                                                                \
    }

#define ROCBLAS_CHECK(err)                                                                                                          \
    {                                                                                                                               \
        rocblas_status err_ = (err);                                                                                                \
        if (err_ != rocblas_status_success)                                                                                         \
        {                                                                                                                           \
            std::cerr << "rocBLAS error: " << rocblas_status_to_string(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                                     \
        }                                                                                                                           \
    }

// Helper to get rocBLAS function
template <typename T>
struct RocblasGemvStridedBatched;

template <>
struct RocblasGemvStridedBatched<float>
{
    static constexpr auto value = rocblas_sgemv_strided_batched;
};
template <>
struct RocblasGemvStridedBatched<double>
{
    static constexpr auto value = rocblas_dgemv_strided_batched;
};
template <>
struct RocblasGemvStridedBatched<rocblas_float_complex>
{
    static constexpr auto value = rocblas_cgemv_strided_batched;
};
template <>
struct RocblasGemvStridedBatched<rocblas_double_complex>
{
    static constexpr auto value = rocblas_zgemv_strided_batched;
};

// Helper functions for CPU verification of rocBLAS complex types
static inline rocblas_float_complex operator+(rocblas_float_complex a, rocblas_float_complex b) { return {std::real(a) + std::real(b), std::imag(a) + std::imag(b)}; }
static inline rocblas_double_complex operator+(rocblas_double_complex a, rocblas_double_complex b) { return {std::real(a) + std::real(b), std::imag(a) + std::imag(b)}; }
static inline rocblas_float_complex operator*(rocblas_float_complex a, rocblas_float_complex b) { return {std::real(a) * std::real(b) - std::imag(a) * std::imag(b), std::real(a) * std::imag(b) + std::imag(a) * std::real(b)}; }
static inline rocblas_double_complex operator*(rocblas_double_complex a, rocblas_double_complex b) { return {std::real(a) * std::real(b) - std::imag(a) * std::imag(b), std::real(a) * std::imag(b) + std::imag(a) * std::real(b)}; }
static inline rocblas_float_complex operator-(rocblas_float_complex a, rocblas_float_complex b) { return {std::real(a) - std::real(b), std::imag(a) - std::imag(b)}; }
static inline rocblas_double_complex operator-(rocblas_double_complex a, rocblas_double_complex b) { return {std::real(a) - std::real(b), std::imag(a) - std::imag(b)}; }
static inline float abs(rocblas_float_complex a) { return std::hypot(std::real(a), std::imag(a)); }
static inline double abs(rocblas_double_complex a) { return std::hypot(std::real(a), std::imag(a)); }
static inline rocblas_float_complex conj(rocblas_float_complex a) { return {std::real(a), -std::imag(a)}; }
static inline rocblas_double_complex conj(rocblas_double_complex a) { return {std::real(a), -std::imag(a)}; }

// CPU verification function
template <typename T>
void cpu_gemv_strided_batched(rocblas_operation trans, int m, int n, const T &alpha, const std::vector<T> &A, const std::vector<T> &x, const T &beta, std::vector<T> &y, int batch_count)
{
    int y_dim = (trans == rocblas_operation_none) ? m : n;
    int dot_dim = (trans == rocblas_operation_none) ? n : m;

    for (int b = 0; b < batch_count; ++b)
    {
        const T *current_A = A.data() + static_cast<size_t>(b) * m * n;
        const T *current_x = x.data() + static_cast<size_t>(b) * dot_dim;
        T *current_y = y.data() + static_cast<size_t>(b) * y_dim;

        // Assuming beta is 0 for this reference as it's 0 in the benchmark
        for (int i = 0; i < y_dim; ++i)
        {
            T sum{};
            if constexpr (std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
            {
                sum = T(0.0, 0.0);
            }
            else
            {
                sum = 0.0;
            }

            for (int j = 0; j < dot_dim; ++j)
            {
                T a_val;
                if (trans == rocblas_operation_none)
                {
                    a_val = current_A[i + static_cast<size_t>(j) * m]; // Access A(i,j) from column-major A
                }
                else
                {
                    a_val = current_A[j + static_cast<size_t>(i) * m]; // Access A(j,i) for transpose
                }

                if constexpr (std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    if (trans == rocblas_operation_conjugate_transpose)
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
    rocblas_operation trans;
    if (trans_str == "N" || trans_str == "n")
        trans = rocblas_operation_none;
    else if (trans_str == "T" || trans_str == "t")
        trans = rocblas_operation_transpose;
    else if (trans_str == "H" || trans_str == "h")
        trans = rocblas_operation_conjugate_transpose;
    else
    {
        std::cerr << "Invalid transpose operation specified. Use N, T, or H." << std::endl;
        exit(EXIT_FAILURE);
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Determine vector sizes based on operation
    int x_dim = (trans == rocblas_operation_none) ? n : m;
    int y_dim = (trans == rocblas_operation_none) ? m : n;

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
        if constexpr (std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
        {
            val = T(dis(gen), dis(gen));
        }
        else
        {
            val = dis(gen);
        }
    }
    for (auto &val : h_x)
    {
        if constexpr (std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
        {
            val = T(dis(gen), dis(gen));
        }
        else
        {
            val = dis(gen);
        }
    }

    // Device data
    T *d_A, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(T)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(T), hipMemcpyHostToDevice));

    // Strides
    rocblas_stride stride_A = static_cast<rocblas_stride>(m) * n;
    rocblas_stride stride_x = x_dim;
    rocblas_stride stride_y = y_dim;

    T alpha, beta;
    if constexpr (std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
    {
        alpha = T(1.0, 0.0);
        beta = T(0.0, 0.0);
    }
    else
    {
        alpha = 1.0;
        beta = 0.0;
    }

    // Warm-up run
    ROCBLAS_CHECK(RocblasGemvStridedBatched<T>::value(handle, trans, m, n, &alpha, d_A, m, stride_A, d_x, 1, stride_x, &beta, d_y, 1, stride_y, batch_count));
    HIP_CHECK(hipDeviceSynchronize());

    // Timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));

    const int num_runs = 100;
    for (int i = 0; i < num_runs; ++i)
    {
        ROCBLAS_CHECK(RocblasGemvStridedBatched<T>::value(handle, trans, m, n, &alpha, d_A, m, stride_A, d_x, 1, stride_x, &beta, d_y, 1, stride_y, batch_count));
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    std::cout << "trans='" << trans_str << "', m=" << m << ", n=" << n << ", batch_count=" << batch_count << std::endl;
    std::cout << "Average execution time: " << milliseconds / num_runs << " ms" << std::endl;

    if (verify)
    {
        std::cout << "\nVerifying result..." << std::endl;
        std::vector<T> h_y_from_gpu(h_y.size());
        HIP_CHECK(hipMemcpy(h_y_from_gpu.data(), d_y, h_y_from_gpu.size() * sizeof(T), hipMemcpyDeviceToHost));

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
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 7)
    {
        std::cerr << "Usage: " << argv[0] << " <trans> <m> <n> <batch_count> <datatype> [verify]" << std::endl;
        std::cerr << "  trans: N (none), T (transpose), H (conjugate transpose)" << std::endl;
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
        run_benchmark<rocblas_float_complex>(trans_str, m, n, batch_count, verify);
    }
    else if (datatype == "z")
    {
        run_benchmark<rocblas_double_complex>(trans_str, m, n, batch_count, verify);
    }
    else
    {
        std::cerr << "Invalid datatype. Use 's', 'd', 'c', or 'z'." << std::endl;
        return 1;
    }

    return 0;
}
