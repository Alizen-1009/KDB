# 如何用 Triton 实现并优化 Softmax？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”Coding 题

## 问题背景

要求实现一个 Triton Softmax；后续整理时应覆盖数值稳定的减最大值、mask、行级并行、block size、访存次数、长行处理，并与 PyTorch reference 做正确性和性能验证。

## 30 秒回答

一行一个 program：按列构造 offset，用 mask 加载并把越界位置置为 `−inf`；减本行最大值、exp、求和、归一化，再按 mask 写回。把 max、exp、sum 和写出融合可避免中间张量往返 HBM。

## 深入解释

行宽设为不小于 N 的 2 的幂，比较 num_warps、寄存器压力和行级并行；特别长的行不能硬塞一个 program，需分块或多阶段实现。

下面是**连续二维张量、每行至少一个有效元素**的基线；代码未在 GPU 上运行。

```python
import triton
import triton.language as tl

@triton.jit
def softmax_rows(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float("inf"))
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    y = e / tl.sum(e, 0)
    tl.store(Y + row * N + col, y, col < N)

def launch_softmax(x, y):
    m, n = x.shape
    assert n > 0 and x.is_contiguous() and y.is_contiguous()
    block = triton.next_power_of_2(n)
    softmax_rows[(m,)](x, y, n, block, num_warps=4)
```

这个版本只适合行宽能容纳在单个 program 资源预算内的情形；大 `N` 要另做分块/在线归约。验证需覆盖非 2 的幂行宽、极大/极小值与不同 dtype，并与 `torch.softmax` 对比。

## 参考来源与待核实

- [Triton Fused Softmax 官方教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
