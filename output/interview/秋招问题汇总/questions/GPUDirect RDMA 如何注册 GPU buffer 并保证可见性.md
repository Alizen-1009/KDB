# GPUDirect RDMA 如何注册 GPU buffer 并保证同步与可见性？

- 整理状态：已整理
- 题目出处：用户提供的 CUDA 截图系列摘要，第 15 题。

## 30 秒回答

GPUDirect RDMA 让支持的第三方设备如 NIC 在平台、驱动和拓扑允许时 DMA 访问经相应机制注册的 GPU 显存，绕过 host 数据缓冲；这与固定 host 页用于异步拷贝不是同一件事。GPU 写完后再让 NIC 读取；NIC 写完后，要按 CUDA 和通信库的同步、内存可见性规则再启动 GPU 消费。DMA 和所有消费者完成前不能注销、释放或复用 buffer。

## 深入解释

注册 GPU buffer 的具体流程依 RDMA 栈、驱动、DMA-BUF 或 peer-memory 等机制及版本而异，不应把 host pinned allocation 当成 GPU 注册。完成通知与 GPU 可见性的保证需查具体 API，不能只靠普通 CPU 标志或假定 kernel 内自动看到 NIC 写入；核对生产者、DMA 和消费者的顺序及生命周期。Managed memory 不可假定适用于同一注册/一致性路径；平台支持、IOMMU、PCIe 拓扑和驱动组合均需验证。

## 关联知识

- [[NVSwitch 和 InfiniBand 上如何设计分层集合通信|多机集合通信]]

## 参考来源与待核实

- [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)：注册、同步和平台限制。具体部署的同步 API 与注册路径待按实际栈核验。

## 所属题单

- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]
- [[../README|秋招问题汇总]]
