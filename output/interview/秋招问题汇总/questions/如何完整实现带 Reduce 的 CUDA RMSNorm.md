# 如何完整实现带 Reduce 的 CUDA RMSNorm？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”Coding 题

## 问题背景

要求写出完整 CUDA 实现，包括沿 hidden dimension 计算平方和的 Reduce、`rsqrt` 归一化、权重缩放、边界处理与 kernel launch；后续整理时应分别验证正确性、数值精度、不同 hidden size 和性能。

## 30 秒回答

每一行先沿 hidden dimension 求 `sum(x²)`，计算 `inv=rsqrt(sum/H+eps)`，再逐元素写 `y=x*inv*weight`。基线实现可让一个 block 处理一行：线程跨步读取并累加 FP32 局部和，先 warp 内归约，再共享内存合并各 warp，广播 inv 后写回。

## 深入解释

要处理 H 非 2 的幂、行 stride、weight 广播和输入/输出 dtype；低精度输入建议 FP32 累加。大 H 或小行数时再考虑多 block/行，避免额外全局归约成本。

下面是面试可写的**连续 FP32、每个 block 一行**版本。`B > 0`、`H > 0`，输入/输出为 `[B,H]`，`weight` 为 `[H]`；代码未在 GPU 上编译或运行。

```cuda
#include <cuda_runtime.h>

__global__ void rmsnorm_f32(const float* x, const float* weight,
                            float* y, int H, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    __shared__ float warp_sum[8];  // launch 使用 256 threads

    float sum = 0.f;
    for (int j = tid; j < H; j += blockDim.x) {
        float v = x[row * H + j];
        sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < 8 ? warp_sum[lane] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) warp_sum[0] = rsqrtf(sum / H + eps);
    }
    __syncthreads();
    float inv = warp_sum[0];
    for (int j = tid; j < H; j += blockDim.x)
        y[row * H + j] = x[row * H + j] * inv * weight[j];
}

void launch_rmsnorm_f32(const float* x, const float* weight, float* y,
                        int B, int H, float eps, cudaStream_t stream) {
    rmsnorm_f32<<<B, 256, 0, stream>>>(x, weight, y, H, eps);
    // 调用方检查 cudaGetLastError()，并在需要时同步 stream。
}
```

验证时与 FP32 reference 对比随机输入、非整除 `H`、不同 `eps`，再测长行和低精度版本；上面只覆盖连续 FP32 基线。

## 参考来源与待核实

- [[../../../../wiki/concepts/RMSNorm|RMSNorm]]

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
