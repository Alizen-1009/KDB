# Triton 在 Ascend 上的支持

## 结论

**Ascend 能运行 Triton 风格 kernel，但要使用专门的 `Triton-Ascend`，不是直接安装上游 NVIDIA/AMD 版 Triton。**

官方项目：[triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend)。它将 Triton 编译栈适配到 CANN、TorchNPU 与 Ascend AI Core。

相关页面：[[../../wiki/concepts/Triton|Triton]]。

## 当前官方支持

核对 Triton-Ascend 官方 `main` commit [`70635d0d`](https://github.com/triton-lang/triton-ascend/commit/70635d0de7e80021a64c70b5e0e29cbc8b44173f)（2026-07-30）：

| 项目 | 当前官方口径 |
| --- | --- |
| 正式版本 | Triton-Ascend 3.2.1 |
| 产品系列 | Atlas A2、A3、950 |
| OS/CPU | Linux aarch64/x86_64 |
| Python | 3.9–3.11 |
| CANN | 推荐 9.0.0 |
| TorchNPU | 2.7.1.post4 |

来源：[README 支持矩阵](https://github.com/triton-lang/triton-ascend/blob/70635d0de7e80021a64c70b5e0e29cbc8b44173f/README.md#L28-L80)。

安装示例：

```bash
# 先按产品与 OS 安装匹配的 CANN / 驱动 / TorchNPU
pip install triton-ascend==3.2.1 \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 简单 kernel 的迁移

Triton kernel body 可以从通用原语开始尝试复用：

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) +
                            tl.load(y + offsets, mask=mask), mask=mask)

x = torch.rand(1024, device="npu")
y = torch.rand(1024, device="npu")
```

host 代码需要把：

```text
device="cuda" → device="npu"
torch.cuda.*   → torch_npu / NPU 对应接口
CUDA stream/event/sync → NPU 对应接口
```

官方迁移指南：[Migrating Triton Operators from GPUs](https://github.com/triton-lang/triton-ascend/blob/70635d0de7e80021a64c70b5e0e29cbc8b44173f/docs/en/migration_guide/migrate_from_gpu.md#L1-L100)。

## 为什么不能认为 CUDA Triton kernel 原样高性能运行

Ascend NPU 与 GPU 的执行模型不同：

- Ascend 使用 AI Core，并区分 Cube Core 与 Vector Core；
- on-chip memory 是 UB/L1，而不是照搬 CUDA shared memory 模型；
- Vector 算子通常要求32-byte访问对齐；Cube-Vector fusion 可能要求512-byte对齐；
- GPU 的超大 logical grid 在 NPU 上可能造成多轮 core launch；
- `coreDim` 不能超过65535；
- `tl.dot` 需要针对 Cube Core 重新选择 M/N/K tile；
- dtype、atomic、mask、非连续访问与低精度矩阵能力不完全一致。

因此通常需要调整：

```text
grid / core count
BLOCK_SIZE
M/N/K tile
UB 占用
数据对齐与 layout
循环和 multi-buffer pipeline
dtype
```

## API 和生态成熟度

项目里程碑记录：

- 2025-06 曾达到约85%的 Triton Python API 覆盖；
- 后续补充 Scan/Sort、非连续访问、atomic、FP8；
- 已适配 vLLM、SGLang、FlagGems 的部分关键 Triton operators。

这表示项目已经可以做实际算子开发，但不能推导为：

- 任意 Triton kernel 都可直接运行；
- 任意 CUDA Triton kernel 都无需修改；
- 整个 vLLM/SGLang 仅安装 Triton-Ascend 就能自动迁移；
- 性能必然接近 AscendC 手写 kernel。

复杂 Attention、MoE、量化和 vendor-specific Triton kernel 仍需逐算子验证。

## 实际判断步骤

1. 确认具体 Ascend/Atlas 产品在支持矩阵内；
2. 对齐驱动、CANN、TorchNPU、Python、Triton-Ascend 版本；
3. 先运行 vector-add 等官方样例；
4. 将 kernel host 侧切换为 NPU tensor/runtime；
5. 检查所用 Triton API 是否已实现；
6. 做 correctness test；
7. 按 AI Core、UB 与对齐要求重新调优；
8. 与 torch_npu、AscendC 或框架原生算子做 benchmark。

## 一句话

> Ascend 可以跑 Triton，但准确说是跑 `Triton-Ascend`。通用 Triton 写法有迁移价值，CUDA/NVIDIA 专属路径不能直接复用，性能优化也必须按 Ascend AI Core 和 UB 重新设计。
