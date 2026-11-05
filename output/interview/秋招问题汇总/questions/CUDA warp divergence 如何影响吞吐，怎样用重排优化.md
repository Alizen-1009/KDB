# CUDA warp divergence 如何影响吞吐，怎样用重排优化？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 1、14 题

## 原题与条件

1. 解释 CUDA 中 warp divergence 的影响及优化方法，并给出一个具体场景的代码示例。
14. 请描述一种减少 warp divergence 的优化方法，并举例说明在分支密集的 CUDA kernel 中如何通过数据预处理或重排来提升性能。

## 30 秒回答

同一 warp 的 lane 走不同路径时，每条路径只让对应 lane 生效。不同 warp 可以走不同路径。短分支考虑 predication，重分支先按类别分组，再发专用 kernel。

## 深入解释

以 ReLU 为例，用 fmaxf 表达单条路径，边界判断通常只影响末尾有效 warp，前提是 block size 为 32 的倍数。分支很重时可以用 flag + prefix sum 生成分组索引，再用 original_index 写回。比较 scan、scatter、launch、访存局部性和端到端收益。

```cpp
__global__ void relu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.0f);
}
```
用 SASS 和 profiler 确认实际分支或谓词执行。

## 关联知识

- [[../../../../wiki/concepts/Warp Divergence|Warp Divergence]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
