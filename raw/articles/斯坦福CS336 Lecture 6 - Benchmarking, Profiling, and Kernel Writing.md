# 斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方课程仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方脚本路径：https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_06.py
- 辅助文件：
  - `lecture_06_utils.py`
  - `gelu.cu`
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：可执行课程讲稿 / 代码

## 原始说明

- 这一讲在官方仓库中主要以可执行 Python 讲稿 `lecture_06.py` 形式存在，而不是单独的 PDF。
- 主题是把 GPU 性能基础推进到“如何实测、如何剖析、如何亲手写 kernel、以及什么时候交给编译器”。
- 讲稿通过 GeLU、softmax、matmul 和 MLP 这些例子，串起 benchmarking、profiling、CUDA、Triton 和 `torch.compile`。

## 讲义结构

### 1. Benchmarking 和 Profiling

- 讲稿强调“不要只看论文和参数表，要 benchmark/profile 自己的代码”。
- Benchmarking 用来看端到端耗时和随规模变化的趋势。
- Profiling 用来看时间到底花在哪些算子和 kernel 上，以及不同输入规模如何触发不同实现路径。
- 讲稿特别强调：GPU benchmark 需要 warmup，并在计时前后显式 `torch.cuda.synchronize()`。

### 2. Kernel Fusion

- 讲稿用 Horace He 的 warehouse/factory 类比解释为什么融合有价值。
- 手写 `manual_gelu` 会拆成多次读写和多个 kernel，而 PyTorch 自带 GeLU 实现往往能更接近融合后的单 kernel 路径。
- 核心原则仍然是：组织计算以最小化 reads/writes。

### 3. CUDA Kernels

- 讲稿展示了如何用 `torch.utils.cpp_extension.load_inline` 把 CUDA/C++ 代码动态编译成 Python 可调用模块。
- 以 GeLU 为例，说明 CUDA kernel 的编程心智是“写单线程逻辑，再通过 grid / block / thread 索引让它并行展开”。
- 对更复杂操作，shared memory、线程协调和数据布局开始变得关键。

### 4. Triton Kernels

- Triton 被定位为比原生 CUDA 更高层的 GPU 编程方式：仍然写 kernel，但以 block 为中心而不是显式操纵线程细节。
- 讲稿展示 Triton GeLU 和 Triton softmax，并通过 PTX 输出观察编译器做了哪些事。
- Triton 的价值不只在“Python 写 kernel”，还在于编译器能自动处理一部分 coalescing、shared memory 和线程粗化等细节。

### 5. torch.compile

- 讲稿把 `torch.compile(manual_gelu)` 和手写 CUDA/Triton 放在同一比较框架中。
- 关键信号是：很多简单融合和代码生成未来会越来越多地由编译器接手，而不需要手写 kernel。
- 但真正理解 profiler、PTX 和内存访问模式，仍然有助于判断自动编译器什么时候有效、什么时候失效。

### 6. Softmax 与 Matmul

- Softmax 被用来说明“聚合型算子”和纯 elementwise kernel 的差异。
- 讲稿用 Triton softmax 解释如何把一整行数据作为一个 block 处理。
- Matmul 部分回到 `tiling + shared memory + L2 cache`，说明矩阵乘法的高性能实现依然围绕数据复用和块级组织。

## 从讲稿中抽出的高信号结论

- 系统优化不是先写 kernel，而是先 benchmark 和 profile，再决定瓶颈究竟在算法、内存访问还是 kernel 实现。
- 手写 CUDA、Triton 和 `torch.compile` 不是互斥路线，而是一条从高层到低层逐级下探的工具链。
- Lecture 5 给的是硬件直觉，Lecture 6 给的是把这些直觉落到日常工程诊断与 kernel 实践中的方法。
