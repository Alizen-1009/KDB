# Grid-stride Loop

## 定义

一种让每个线程以 `gridDim * blockDim` 为步长处理多个元素的遍历模式，用来覆盖大于当前 launch 配置的数据范围。

## 它解决什么问题

- 让 kernel 在固定 launch 配置下处理任意长度输入
- 避免只覆盖一轮索引导致的越界或遗漏
- 在 `softmax`、[[Histogram]]、向量归约等场景里提供稳定的并行遍历骨架

## 核心机制

- 先计算线程的全局起始索引
- 每轮处理当前位置上的元素
- 处理完后按 `gridDim.x * blockDim.x` 或 `blockDim.x` 递增，继续处理下一项
- 配合边界判断保证最后一轮不越界

## 关键权衡

- 写法简单、适用范围广，但并不自动保证最优访存模式
- 一维 grid-stride 常见于向量和直方图；行内遍历版本常见于 `softmax`、`norm` 这类“每个 block 对应一行”的 kernel
- 如果每次迭代工作量不均衡，仍可能出现负载不平衡

## 相关来源

- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[CUDA Kernel]]
- [[Block Reduce]]
- [[Histogram]]

## 研究备注

- 这类循环是很多 CUDA 题的第一层骨架；面试里往往默认你会写，真正拉开差距的是你能否进一步说明 coalescing 和 reduce 组织方式
