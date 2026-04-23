# Tail Effect

## 定义

`Tail effect` 指 kernel 执行到最后几轮 block 时，由于剩余 block 数不足以填满所有 SM，导致硬件利用率明显下降的现象。

## 它解决什么问题

- 解释为什么某些 kernel 的平均吞吐不差，但整体尾部阶段仍然拖慢总时延
- 帮助分析 launch 配置是否让 block 数量充分覆盖 SM

## 核心机制

- GPU 通常通过大量 block 分发到多个 SM 上保持并发
- 当剩余 block 太少时，最后一波只能占用部分 SM
- 如果 block 太大、block 数太少或问题规模不合适，就更容易出现 tail effect

## 关键权衡

- 增加 block 数有助于改善尾部利用率，但过小 block 也可能降低单 block 效率
- launch 配置要同时平衡单 block 工作量、occupancy 和 SM 覆盖率

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]

## 相关概念

- [[CUDA Kernel]]
- [[GPU执行模型]]
- [[Occupancy]]

## 研究备注

- 后续可补不同 block size / grid size 对 tail effect 的 profiler 截图示例
