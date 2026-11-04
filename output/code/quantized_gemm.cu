// Build: nvcc -O3 -std=c++17 -arch=sm_89 quantized_gemm.cu -o quantized_gemm
// Inputs: row-major A[M,K] (FP16), W[N,K] (FP32), finite values.
// Output: C[M,N] (FP32) = A * transpose(W), with quantized operands.
// W4 uses signed nibbles (even k in the low nibble), K groups of 32/64/128.
// W8 uses one scale per output channel. A8 uses one dynamic scale per row.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                     cudaGetErrorString(error));                          \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

constexpr int TILE = 16;
constexpr int QUANT_THREADS = 128;

enum class Format { W4A16, W8A8, W4A8 };

template <typename T>
__device__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float to_float(__half value) {
    return __half2float(value);
}

__device__ int quantize_value(float value, float scale, int limit) {
    // Divide directly so a small scale need not have a representable reciprocal.
    return __float2int_rn(fminf(float(limit), fmaxf(float(-limit), value / scale)));
}

template <typename Input, bool Packed>
__global__ void quantize_rows(const Input* input, void* quantized, float* scales,
                              int k_size, int group_size, int groups) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int start = group * group_size;
    const int length = min(group_size, k_size - start);
    const Input* values = input + size_t(row) * k_size + start;
    constexpr int limit = Packed ? 7 : 127;
    __shared__ float maxima[QUANT_THREADS];

    float peak = 0.0f;
    for (int k = threadIdx.x; k < length; k += QUANT_THREADS) {
        peak = fmaxf(peak, fabsf(to_float(values[k])));
    }
    maxima[threadIdx.x] = peak;
    __syncthreads();
    for (int stride = QUANT_THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = maxima[0] == 0.0f ? 1.0f : fmaxf(maxima[0] / limit, 0x1p-149f);
    if (threadIdx.x == 0) {
        scales[size_t(row) * groups + group] = scale;
    }

    if constexpr (Packed) {
        auto* bytes = static_cast<uint8_t*>(quantized)
                    + size_t(row) * ((k_size + 1) / 2) + start / 2;
        for (int i = threadIdx.x; i < (length + 1) / 2; i += QUANT_THREADS) {
            const int k = 2 * i;
            const int low = quantize_value(to_float(values[k]), scale, limit);
            const int high = k + 1 < length
                           ? quantize_value(to_float(values[k + 1]), scale, limit) : 0;
            bytes[i] = uint8_t((low & 15) | ((high & 15) << 4));
        }
    } else {
        auto* codes = static_cast<int8_t*>(quantized) + size_t(row) * k_size + start;
        for (int k = threadIdx.x; k < length; k += QUANT_THREADS) {
            codes[k] = int8_t(quantize_value(to_float(values[k]), scale, limit));
        }
    }
}

template <bool Packed>
__device__ int weight_code(const void* weights, int n, int k, int k_size) {
    if constexpr (Packed) {
        const auto* bytes = static_cast<const uint8_t*>(weights);
        const int byte = bytes[size_t(n) * ((k_size + 1) / 2) + k / 2];
        const int nibble = (byte >> (4 * (k & 1))) & 15;
        return (nibble ^ 8) - 8;
    } else {
        return static_cast<const int8_t*>(weights)[size_t(n) * k_size + k];
    }
}

template <bool Packed, bool A8>
__global__ void gemm_kernel(const void* activations, const void* weights,
                            const float* a_scales, const float* w_scales, float* output,
                            int m_size, int n_size, int k_size, int group_size, int groups) {
    using Element = typename std::conditional<A8, int8_t, float>::type;
    using Partial = typename std::conditional<A8, int32_t, float>::type;
    __shared__ Element a_tile[TILE][TILE];
    __shared__ Element w_tile[TILE][TILE + 1];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int m = blockIdx.y * TILE + y;
    const int n = blockIdx.x * TILE + x;
    const int load_n = blockIdx.x * TILE + y;
    const float a_scale = A8 && m < m_size ? a_scales[m] : 1.0f;
    float result = 0.0f;

    for (int group = 0; group < groups; ++group) {
        const int start = group * group_size;
        const int end = start + min(group_size, k_size - start);
        Partial partial = 0;
        for (int base = start; base < end; base += TILE) {
            const int k = base + x;
            Element a = 0;
            if (m < m_size && k < end) {
                const size_t index = size_t(m) * k_size + k;
                if constexpr (A8) {
                    a = static_cast<const int8_t*>(activations)[index];
                } else {
                    a = __half2float(static_cast<const __half*>(activations)[index]);
                }
            }
            a_tile[y][x] = a;
            w_tile[y][x] = load_n < n_size && k < end
                         ? Element(weight_code<Packed>(weights, load_n, k, k_size)) : 0;
            __syncthreads();

            #pragma unroll
            for (int t = 0; t < TILE; ++t) {
                if constexpr (A8) {
                    partial += int(a_tile[y][t]) * int(w_tile[x][t]);
                } else {
                    partial = fmaf(a_tile[y][t], w_tile[x][t], partial);
                }
            }
            __syncthreads();
        }
        if (m < m_size && n < n_size) {
            // A W4 group has its own scale, so dequantize before adding it to C.
            result += (float(partial) * w_scales[size_t(n) * groups + group]) * a_scale;
        }
    }
    if (m < m_size && n < n_size) {
        output[size_t(m) * n_size + n] = result;
    }
}

// Weight quantization can run once and its outputs can be reused across calls.
// For W4 allocate N*ceil(K/2) bytes and N*ceil(K/group_size) floats.
// For W8 allocate N*K bytes and N floats.
cudaError_t quantize_weights(Format format, const float* w, void* codes, float* scales,
                             int n, int k, int group_size = 128, cudaStream_t stream = nullptr) {
    if (n <= 0 || k <= 0) return cudaErrorInvalidValue;
    if (format == Format::W8A8) {
        quantize_rows<float, false><<<n, QUANT_THREADS, 0, stream>>>(w, codes, scales, k, k, 1);
    } else {
        if (group_size != 32 && group_size != 64 && group_size != 128) return cudaErrorInvalidValue;
        const int groups = 1 + (k - 1) / group_size;
        if (groups > 65535) return cudaErrorInvalidValue;
        quantize_rows<float, true><<<dim3(n, groups), QUANT_THREADS, 0, stream>>>(
            w, codes, scales, k, group_size, groups);
    }
    return cudaGetLastError();
}

// A8 requires caller-owned scratch (M*K bytes and M floats).
// Scratch may be reused after the stream reaches the end of this operation.
cudaError_t quantized_gemm(Format format, const __half* a, const void* w_codes,
                           const float* w_scales, float* c, int m, int n, int k,
                           int8_t* a_codes = nullptr, float* a_scales = nullptr,
                           int group_size = 128, cudaStream_t stream = nullptr) {
    if (m <= 0 || n <= 0 || k <= 0 || (m - 1) / TILE >= 65535) return cudaErrorInvalidValue;
    const bool packed = format != Format::W8A8;
    if (packed) {
        if (group_size != 32 && group_size != 64 && group_size != 128) return cudaErrorInvalidValue;
    } else {
        group_size = k;
        // Worst-case signed INT8 product is 127*127 for this symmetric quantizer.
        if (k > INT32_MAX / (127 * 127)) return cudaErrorInvalidValue;
    }
    const int groups = 1 + (k - 1) / group_size;
    if (groups > 65535) return cudaErrorInvalidValue;
    const dim3 grid(1 + (n - 1) / TILE, 1 + (m - 1) / TILE);
    const dim3 block(TILE, TILE);

    if (format == Format::W4A16) {
        gemm_kernel<true, false><<<grid, block, 0, stream>>>(
            a, w_codes, nullptr, w_scales, c, m, n, k, group_size, groups);
    } else {
        if (a_codes == nullptr || a_scales == nullptr) return cudaErrorInvalidValue;
        quantize_rows<__half, false><<<m, QUANT_THREADS, 0, stream>>>(
            a, a_codes, a_scales, k, k, 1);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return error;
        if (packed) {
            gemm_kernel<true, true><<<grid, block, 0, stream>>>(
                a_codes, w_codes, a_scales, w_scales, c, m, n, k, group_size, groups);
        } else {
            gemm_kernel<false, true><<<grid, block, 0, stream>>>(
                a_codes, w_codes, a_scales, w_scales, c, m, n, k, k, 1);
        }
    }
    return cudaGetLastError();
}

#ifndef QUANTIZED_GEMM_NO_MAIN
template <typename T>
T* device_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

// Independent CPU reference: explicitly dequantize each operand before its dot product.
static void reference_quantize(float* values, int length, int limit) {
    float peak = 0.0f;
    for (int k = 0; k < length; ++k) peak = std::max(peak, std::abs(values[k]));
    const float scale = peak == 0.0f ? 1.0f : peak / limit;
    for (int k = 0; k < length; ++k) values[k] = std::nearbyint(values[k] / scale) * scale;
}

int main() {
    constexpr int M = 19, N = 23, K = 131, GROUP = 64;
    std::vector<__half> a(M * K);
    std::vector<float> w(N * K), output(M * N);
    for (int i = 0; i < M * K; ++i) a[i] = __float2half(i < K ? 0.0f : std::sin(i * 0.13f));
    for (int i = 0; i < N * K; ++i) {
        w[i] = i < GROUP ? 0.0f : std::cos(i * 0.17f) * (1 + (i % K) / GROUP);
    }
    auto* a_device = device_alloc<__half>(a.size());
    auto* w_device = device_alloc<float>(w.size());
    auto* w_codes = device_alloc<uint8_t>(w.size());
    auto* w_scales = device_alloc<float>(N * ((K + GROUP - 1) / GROUP));
    auto* a_codes = device_alloc<int8_t>(a.size());
    auto* a_scales = device_alloc<float>(M);
    auto* c_device = device_alloc<float>(output.size());
    CUDA_CHECK(cudaMemcpy(a_device, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w_device, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    bool passed = true;
    for (Format format : {Format::W4A16, Format::W8A8, Format::W4A8}) {
        const char* name = format == Format::W4A16 ? "W4A16"
                         : format == Format::W8A8 ? "W8A8" : "W4A8";
        CUDA_CHECK(quantize_weights(format, w_device, w_codes, w_scales, N, K, GROUP));
        CUDA_CHECK(quantized_gemm(format, a_device, w_codes, w_scales, c_device,
                                  M, N, K, a_codes, a_scales, GROUP));
        CUDA_CHECK(cudaMemcpy(output.data(), c_device, output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        std::vector<float> a_ref(a.size()), w_ref = w;
        for (size_t i = 0; i < a.size(); ++i) a_ref[i] = __half2float(a[i]);
        if (format != Format::W4A16) {
            for (int m = 0; m < M; ++m) reference_quantize(a_ref.data() + m * K, K, 127);
        }
        const int group = format == Format::W8A8 ? K : GROUP;
        const int limit = format == Format::W8A8 ? 127 : 7;
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; k += group) {
                reference_quantize(w_ref.data() + n * K + k, std::min(group, K - k), limit);
            }
        }

        float max_error = 0.0f;
        bool matches = true;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected = std::fma(a_ref[m * K + k], w_ref[n * K + k], expected);
                }
                const float error = std::abs(output[m * N + n] - expected);
                max_error = std::max(max_error, error);
                matches &= error <= 2e-4f + 1e-5f * std::abs(expected);
            }
        }
        std::printf("%s %s, max absolute error %.6g\n", name, matches ? "PASS" : "FAIL", max_error);
        passed &= matches;
    }

    for (void* ptr : {static_cast<void*>(a_device), static_cast<void*>(w_device),
                      static_cast<void*>(w_codes), static_cast<void*>(w_scales),
                      static_cast<void*>(a_codes), static_cast<void*>(a_scales),
                      static_cast<void*>(c_device)}) {
        CUDA_CHECK(cudaFree(ptr));
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
