# 如何并行计算大量 bounding box 的 IoU？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

先确认是逐对、全量两两，还是阈值筛选/NMS。轴对齐 xyxy 框的交集宽高取 max(0,min(right)-max(left))，IoU=交集/(areaA+areaB−交集)。逐对每线程一对；全量两两用 2D tile，复用两组 box。

## 深入解释

统一连续/像素坐标约定，定义并集为零的返回策略，面积和除法用 FP32。N×M 全量输出有 Ω(NM) 写流量，不能靠 tiling 降低输出阶数。预计算面积，按 warp 合并写，box 可用 SoA 或对齐 float4；小 tile 先测缓存版，再测 shared staging。阈值 τ≥0 且并集>0 时可比较 intersection 与 τ×union 省除法；只要匹配时可做空间分桶、粗筛和稀疏输出。NMS 本身有分数顺序依赖，可并行算重叠掩码，再按顺序处理抑制。测试无交、包含、重合、退化和坐标边界。

## 逐对计算的 CUDA 示例

每个线程计算一对，输入为有效且有限的连续坐标 `(x1, y1, x2, y2)`。全量矩阵时可将 `a[i]`、`b[i]` 换成按二维输出索引加载，并让相邻 lane 写相邻的列。

```cpp
__global__ void paired_iou(const float4* a, const float4* b,
                           float* out, std::size_t n) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 x = a[i], y = b[i];
    const float w = fmaxf(0.0f, fminf(x.z, y.z) - fmaxf(x.x, y.x));
    const float h = fmaxf(0.0f, fminf(x.w, y.w) - fmaxf(x.y, y.y));
    const float inter = w * h;
    const float union_area = (x.z - x.x) * (x.w - x.y)
                           + (y.z - y.x) * (y.w - y.y) - inter;
    out[i] = union_area > 0.0f ? inter / union_area : 0.0f;
}
```

示例的退化并集约定为 0，未运行 GPU 测试。

## 关联知识

- [[../../../../wiki/concepts/内存合并访问|内存合并访问]]

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
