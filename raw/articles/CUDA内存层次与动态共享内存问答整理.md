# CUDA内存层次与动态共享内存问答整理

## 来源信息

- 类型：对话整理 / CUDA 基础笔记
- 日期：2026-05-06
- 主题：CUDA kernel launch 参数、内存层次、SM 示意图、occupancy、动态 shared memory
- 适用范围：面试复习、CUDA kernel 入门、性能分析基础

## Kernel launch 第三个参数

CUDA kernel launch 常见形式是：

```cpp
kernel<<<blocks_in_grid, threads_in_block, shared_mem_bytes, stream>>>(args...);
```

第三个参数 `shared_mem_bytes` 表示每个 block 额外申请的 **dynamic shared memory** 大小，单位是 byte。

例如：

```cpp
__global__ void kernel(float* x) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    smem[tid] = x[tid];
}

kernel<<<1, 256, 256 * sizeof(float)>>>(x);
```

这里第三个参数表示每个 block 申请 `256 * sizeof(float)` byte 的动态 shared memory。

如果写成：

```cpp
kernel<<<blocks_in_grid, threads_in_block, 0, stream>>>(args...);
```

则表示不额外申请动态 shared memory，并且 kernel 在指定 `stream` 上执行。

## 静态 shared memory 与动态 shared memory

静态 shared memory 的大小在编译期确定，不依赖 kernel launch 第三个参数：

```cpp
__global__ void kernel(float* x) {
    __shared__ float buf[256];
}
```

动态 shared memory 使用 `extern __shared__` 声明，大小由 launch 第三个参数决定：

```cpp
__global__ void kernel(float* x) {
    extern __shared__ float buf[];
}
```

如果 kernel 中访问了 `extern __shared__`，但 launch 时不传第三个参数，等价于传 `0`：

```cpp
kernel<<<grid, block>>>(args);
kernel<<<grid, block, 0>>>(args);
```

CUDA 不会根据 `extern __shared__` 自动推断需要多少空间，因为声明里没有长度信息。若未申请或申请不足却访问 `buf[tid]`，就是越界访问，结果属于未定义行为，可能表现为结果错误、kernel illegal memory access，或在某些情况下看似正常但结果不稳定。

当静态和动态 shared memory 同时存在时，每个 block 的 shared memory 总量是二者之和：

```cpp
__global__ void kernel() {
    __shared__ float fixed[256];   // 静态 shared memory
    extern __shared__ float dyn[]; // 动态 shared memory
}

kernel<<<grid, block, 2048>>>();
```

这里每个 block 至少使用 `256 * sizeof(float) + 2048` byte shared memory。

## CUDA 内存层次

CUDA 内存层次可以按速度、容量和可见范围理解：

```text
Register
  ↓
Shared Memory / L1 Cache
  ↓
L2 Cache
  ↓
Global Memory
  ↓
Host Memory
```

### Register

寄存器是每个 thread 私有的最快存储。普通局部标量变量通常会被编译器放在寄存器中。

特点：

- 每个 thread 私有
- 速度最快，容量最小
- 使用量由编译器分配和代码形态共同决定
- 每个 thread 使用太多寄存器会降低 SM 上可驻留 thread / warp 数，进而降低 occupancy
- 如果寄存器不够，变量可能 spill 到 local memory

### Local Memory

Local memory 名字容易误导。它是每个 thread 私有的地址空间，但物理上通常位于 global memory，并通过 cache 访问。

常见触发场景：

- 寄存器 spill
- 局部数组太大
- 编译器无法静态确定索引的局部数组

因此 local memory 不等于“很近的本地内存”，性能上通常要当作慢路径看待。

### Shared Memory

Shared memory 是每个 block 内线程共享的片上内存，适合显式做数据复用。

特点：

- block 内所有 thread 可见
- 不同 block 之间不可见
- 速度很快，但容量有限
- 常用于 tiling、block 内 reduce、矩阵乘法 tile 缓存、数据重排
- 需要注意 bank conflict

### L1 Cache

L1 cache 通常位于每个 SM 附近，用于缓存 global/local memory 访问。它由硬件管理，程序员一般不能像 shared memory 那样显式索引。可以粗略理解为：shared memory 是程序员显式管理的片上缓存，L1 是硬件自动管理的片上缓存。

### L2 Cache

L2 cache 通常由整个 GPU 共享。global memory 访问、跨 SM 数据复用和一部分原子操作路径会经过 L2。它比 L1 慢，但比显存快。

### Global Memory

Global memory 是 GPU 显存中的主要大容量内存，所有 thread 和 block 都可以访问。

特点：

- 容量最大
- 延迟高
- 带宽高，但需要 coalesced access 才能高效利用
- 是 CUDA kernel 中最常见的性能瓶颈之一

典型优化方向是让同一个 warp 中连续 thread 访问连续、对齐的地址。

### Constant Memory 与 Texture / Read-only Cache

Constant memory 适合存放所有 thread 读取相同或少量常量数据。若一个 warp 中 thread 读取同一个 constant 地址，访问可以很高效；若读取地址分散，性能会下降。

Texture memory / read-only cache 适合某些只读数据访问模式，尤其是带空间局部性的图像或二维访问场景。现代 CUDA 中很多场景可以用普通 global load 或 read-only path 替代，但理解其只读缓存属性仍有帮助。

### Host Memory

Host memory 是 CPU 侧内存。GPU 访问 host memory 通常比访问 device global memory 慢。`cudaMallocHost` 分配的 pinned memory 更适合 H2D / D2H 高速拷贝和异步传输。

## SM 结构示意图如何理解

常见 SM 示意图会画出 warp scheduler、dispatch unit、register file、INT32 / FP32 / FP64 执行单元、LD/ST、SFU、Tensor Core，以及 L1 data cache / shared memory。

这类图适合理解 SM 内部执行资源，但不应直接当作所有 GPU 架构通用的 CUDA memory hierarchy 图。

需要注意：

- 不同 GPU 架构的 SM 结构、shared memory 容量、Tensor Core 数量和 FP64 配比不同
- 图中 `Warp Scheduler (32 thread/clk)` 通常是在表达一次调度一个 warp，但真实吞吐取决于指令类型、依赖关系、执行单元可用性、memory stall 和架构细节
- Register file 是硬件资源，程序员不能手动选择使用哪一块 register file，只能通过代码、编译选项和 occupancy 分析间接影响寄存器使用
- L1 data cache / shared memory 是 SM 级片上资源，不是每个 warp scheduler 独占一份

## Occupancy

`Occupancy` 指一个 SM 上实际活跃 warp 数占该 SM 理论最大活跃 warp 数的比例。

```text
occupancy = active warps per SM / max warps per SM
```

例如，一个 SM 最多可驻留 `64` 个 warp。如果某个 kernel 由于资源限制实际只能驻留 `32` 个 warp，则 occupancy 为 `50%`。

这里的“驻留”不是说所有 warp 在同一瞬间执行，而是说这些 warp 的上下文已经在 SM 上，warp scheduler 可以在它们之间切换。

寄存器和 shared memory 会影响 occupancy，因为每个 SM 的资源有限：

```text
每个 SM 有固定数量 registers
每个 SM 有固定容量 shared memory
每个 SM 有固定最大 blocks / warps / threads
```

如果每个 thread 使用太多寄存器，一个 SM 能放下的 thread / block 数会减少。如果每个 block 使用太多 shared memory，一个 SM 能放下的 block 数也会减少。

例子：

```text
SM 有 65536 个 32-bit registers
block 有 256 threads

每 thread 32 registers:
每 block 需要 256 * 32 = 8192 registers

每 thread 128 registers:
每 block 需要 256 * 128 = 32768 registers
```

后者会显著减少每个 SM 能驻留的 block 数。

Occupancy 的意义在于隐藏延迟。当一个 warp 等待 global memory 或长延迟指令时，scheduler 可以切换到其他 ready warp。若 occupancy 太低，可切换 warp 太少，SM 更容易空等。

但 occupancy 不是越高越好。某些 kernel 虽然 occupancy 不满，但因为寄存器复用好、shared memory 减少了 global memory 访问、ILP 高或计算密度高，仍然可能很快。性能分析时需要结合实际瓶颈，而不是单独追求 100% occupancy。

## 一句话总结

- Kernel launch 第三个参数只控制动态 shared memory，默认是 `0`，CUDA 不会自动推断
- `__shared__ T buf[N]` 是静态 shared memory，编译期确定大小
- `extern __shared__ T buf[]` 是动态 shared memory，launch 时必须给足空间
- CUDA 内存层次里 register 最快但最少，shared memory 快且可控，global memory 大但慢
- 寄存器和 shared memory 用太多会降低 occupancy，但最高 occupancy 不一定对应最高性能
