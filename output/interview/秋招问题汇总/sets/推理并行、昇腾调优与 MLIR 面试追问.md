# 推理并行、昇腾调优与 MLIR 面试追问

复习优先从 PCP 与 DCP 的适用场景开始。随后用异构 PD、算子 A/B 和 SFT 配置练习如何把机制与证据讲清楚。Linear、DCP 内核和 MLIR 部分适合作为源码追问。

## PCP、DCP、异构 PD 与性能验证

- [[../questions/PCP 与 DCP 各解决什么瓶颈，什么时候开|PCP 与 DCP 各解决什么瓶颈，什么时候开]]
- [[../questions/PD 分离中为什么用国产卡做 Prefill、NVIDIA 做 Decode|PD 分离中为什么用国产卡做 Prefill、NVIDIA 做 Decode]]
- [[../questions/昇腾 A5 相比 A3 升级在哪里，FP16 和 FP8 峰值是多少|昇腾 A5 相比 A3 升级在哪里，FP16 和 FP8 峰值是多少]]
- [[../questions/固定 shape 下如何制定算子调优策略|固定 shape 下如何制定算子调优策略]]
- [[../questions/如何估算推理系统理论上限，并定义 SLA 下的最大吞吐|如何估算推理系统理论上限，并定义 SLA 下的最大吞吐]]

## 训练与 SFT

- [[../questions/SFT 如何构建数据并验证调参效果|SFT 如何构建数据并验证调参效果]]
- [[../questions/SFT 如何说明训练框架和 Base Model，怎样判断原生多模态|SFT 如何说明训练框架和 Base Model，怎样判断原生多模态]]
- [[../questions/NCCL 与 Gloo 等分布式通信后端的优劣和适用场景是什么|NCCL 与 Gloo 等分布式通信后端的优劣和适用场景是什么]]
- [[../questions/TP8、PP4、DP2 下全局 batch 如何计算|TP8、PP4、DP2 下全局 batch 如何计算]]

## Linear、Decode TopK 与 GPU 同步

- [[../questions/W8A8 Triton、Marlin FP8 与 Persistent Matmul 是什么关系|W8A8 Triton、Marlin FP8 与 Persistent Matmul 是什么关系]]
- [[../questions/寄存器与共享内存如何影响 occupancy，Persistent Kernel 如何隐藏延迟|寄存器与共享内存如何影响 occupancy，Persistent Kernel 如何隐藏延迟]]
- [[../questions/稀疏 Attention 的 Decode TopK 后端如何选择|稀疏 Attention 的 Decode TopK 后端如何选择]]
- [[../questions/DCP 稀疏索引如何选全局 TopK，两个 pack 有何区别|DCP 稀疏索引如何选全局 TopK，两个 pack 有何区别]]
- [[../questions/Kernel 内同步、Kernel 间数据交接和跨卡通信有何区别|Kernel 内同步、Kernel 间数据交接和跨卡通信有何区别]]
- [[../questions/DCP Attention 的 AG 加 RS 与 A2A 合并有何区别|DCP Attention 的 AG 加 RS 与 A2A 合并有何区别]]

## CMake、TableGen 与 MLIR

- [[../questions/CMake、Ninja 与 g++ 如何分工|CMake、Ninja 与 g++ 如何分工]]
- [[../questions/MLIR Op 如何从 TableGen 生成 C++ 并注册到 Dialect|MLIR Op 如何从 TableGen 生成 C++ 并注册到 Dialect]]
- [[../questions/MLIR ODS 合同、类型推导和 Verifier 如何分工|MLIR ODS 合同、类型推导和 Verifier 如何分工]]
- [[../questions/grep -E 如何搜索构建文件中的多种模式|grep -E 如何搜索构建文件中的多种模式]]

[[../README|返回秋招问题汇总]]
