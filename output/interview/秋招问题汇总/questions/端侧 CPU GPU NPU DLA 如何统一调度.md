# 端侧 CPU、GPU、NPU、DLA 如何统一调度？

- 整理状态：已整理
- 题目出处：用户提供的“小鹏汽车三面｜端侧 AI Infra 面经”

## 30 秒回答

先把每帧任务表示为带依赖和截止时间的 DAG，把每个算子在各硬件上的可执行性、shape/精度支持、实测时延和功耗做成 capability 表；调度时一起计算算子执行、跨设备搬运、格式转换与同步成本。优先保证关键路径上的帧截止时间和正确性，再用可并行的非关键任务填补设备空闲。运行时需要明确每条边的 buffer 所有权和完成事件，并监控 P99 帧时延、丢帧、设备利用率与功耗；不能只把 GPU 利用率最大化当目标。

## 深入解释

1. **图与能力**：算子节点标注输入输出 shape、dtype、layout、截止时间；候选执行器记录 CPU/GPU/NPU/DLA 是否支持、是否需要编译或 fallback、该 shape 的延迟与资源需求。DLA 是受支持层和格式限制的专用执行器，不能把任意 CUDA 算子直接派过去。
2. **边成本**：对跨硬件数据流记录共享内存是否真正可用、是否需要缓存同步、拷贝、重排、量化/反量化和 fence。即便地址共享，也可能有内存带宽与同步开销。
3. **调度目标**：先满足正确性、帧截止时间和温度/功耗约束；按关键路径、预计完成时间和设备队列做放置与背压。可用离线 profiling 建初始表，运行时按负载更新估计并留安全余量。
4. **落地**：统一的是算子接口、tensor 描述、事件和错误回退，不要求所有硬件具有同一指令或内存语义。实车场景还应对超时、设备不可用和版本不兼容定义降级路径。

## 面试官可能追问

- 某节点在 GPU 上算得快，但两侧都在 NPU 上时，为什么整体反而可能慢？
- 如何防止跨帧重叠时读写同一 buffer？
- 设备利用率很高但仍丢帧，下一步看什么？

## 关联知识

- [[跨帧流水线为何产生读写竞争，三种缓冲方案如何取舍|跨帧流水线缓冲取舍]]
- [[../../../../wiki/concepts/算子融合|算子融合]]
- [[../../../../wiki/concepts/Benchmarking|Benchmarking]]

## 参考来源与待核实

- [NVIDIA TensorRT：Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)：DLA 的层、格式及版本边界；其他 NPU 与自研芯片需查对应平台文档。
- [NVIDIA CUDA Programming Guide：异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：stream/event 同步机制示例。实际 CPU/GPU/NPU/DLA 跨设备同步需按运行时 API 验证。
- 这里是通用架构设计答案；用户转述的团队硬件、框架现状与利用率数字未经独立核验，不作为本页性能事实。

## 所属题单

- [[../sets/小鹏汽车端侧AI Infra三面|小鹏汽车端侧 AI Infra 三面]]
- [[../sets/平台与工程|平台与工程]]
- [[../README|秋招问题汇总]]
