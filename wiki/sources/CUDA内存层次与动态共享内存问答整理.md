# CUDA内存层次与动态共享内存问答整理

## 来源信息

- 标题：CUDA内存层次与动态共享内存问答整理
- 作者：对话整理
- 日期：2026-05-06
- 类型：文章 / 对话整理
- 原始文件：`raw/articles/CUDA内存层次与动态共享内存问答整理.md`

## 2-3 条核心摘要

- 这份资料解释了 CUDA kernel launch 第三个参数：它控制每个 block 的动态 shared memory 大小，单位是 byte；不传时默认是 `0`，CUDA 不会根据 `extern __shared__` 自动推断需要的空间。
- 资料系统梳理了 CUDA 内存层次：register 是 thread 私有最快存储，local memory 常来自 spill 且通常落在 global memory 路径上，shared memory 是 block 内共享片上内存，L1/L2 是硬件 cache，global memory 容量大但延迟高。
- 资料补充了 occupancy 的直觉：它衡量每个 SM 上实际驻留 active warps 占最大 warps 的比例；寄存器和 shared memory 用量会限制每个 SM 可驻留 block / warp 数，但最高 occupancy 不必然等于最高性能。

## 值得关注的论断

- `extern __shared__` 只声明一段动态 shared memory，不包含长度信息；如果 launch 时没有申请或申请不足却访问它，就是越界访问，属于未定义行为。
- 静态 shared memory 和动态 shared memory 可以同时存在；每个 block 的 shared memory 总占用是二者之和，这会进一步影响 occupancy。
- SM 结构示意图适合理解 warp scheduler、register file、执行单元和 L1/shared memory 等硬件资源，但不能当作所有 GPU 架构通用的 CUDA 内存层次图。

## 关键概念

- [[CUDA内存层次]]
- [[动态共享内存]]
- [[CUDA Kernel]]
- [[GPU执行模型]]
- [[Occupancy]]
- [[Bank Conflict]]
- [[内存合并访问]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 创建概念页：`CUDA内存层次`、`动态共享内存`
- 更新概念页：`CUDA Kernel`、`GPU执行模型`、`Occupancy`、`Bank Conflict`、`内存合并访问`
- 是否存在冲突：未发现与现有 wiki 的直接冲突；本资料主要补足 CUDA memory hierarchy 与 launch-time dynamic shared memory 这条基础线索。

## 待确认

- SM 示意图中具体数值如 register file 大小、L1/shared memory 容量和 Tensor Core 数量依赖 GPU 架构；当前页只记录通用理解，后续若引入 A100 / H100 / Blackwell 具体资料，应拆到硬件实体页或架构页中。
