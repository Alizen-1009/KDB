# NVLink 多 GPU 如何做 P2P 并安排通信？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 8、15 题

## 原题与条件

8. 在 NVLink 互连的 multi-GPU 系统中，如何使用 CUDA 感知的 MPI 或 NCCL 实现高效的点对点通信？请对比与 PCIe 通信的差异。
15. 在 NVLink 多 GPU 系统中，如何设计数据分布与通信策略以最小化跨 GPU 传输开销？请结合 peer-to-peer 访问和集体通信原语说明。

## 30 秒回答

先查拓扑和 peer 可达性。单进程复制可用 cudaMemcpyPeerAsync，多 rank 用匹配的 ncclSend/Recv，或直接给 CUDA-aware MPI 传 device pointer。实际路径取决于拓扑和实现。

## 深入解释

用 nvidia-smi topo -m、cudaDeviceCanAccessPeer 和微基准验证。互相依赖的收发放进 NCCL group。PCIe 也可支持 P2P；测小消息时延、大消息单/双向带宽。把高频 TP 放同一 NVLink 域，PP 邻接点用 P2P，输出可保持分片就选 ReduceScatter；需要全量再 AllGather。MoE EP 按目的 rank 交换 token，需考虑热点。

截图系列追问：UVA 只统一虚拟地址表示，不代表 GPU 间一定可直接访问。先对访问方 A 查询 `cudaDeviceCanAccessPeer(A,B)`，切到 A 后对 B 调用 `cudaDeviceEnablePeerAccess(B,0)`；反向访问须分别查询和启用。生产者完成后用 event 等正确建立依赖，并在消费者完成前保留 buffer；设备、平台、分配方式及拓扑限制都要核对。即使有 NVLink，也需验证实际 P2P 能力与路径。

## 关联知识

- [[../../../../wiki/concepts/集合通信|集合通信]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)
- [CUDA Programming Guide：Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)。

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]（用户提供的截图系列摘要，第 14 题）
- [[../README|秋招问题汇总]]
