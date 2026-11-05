# ZeRO 与 FSDP 是什么关系，有什么区别？

- 整理状态：已整理
- 题目出处：既有面试原稿与用户本次提供的 FSDP 前反向题

## 其他问法

- FSDP 怎么实现，前向和反向各有什么通信？

## 30 秒回答

ZeRO 按 stage 定义训练状态的分片范围。FSDP1 的 FULL_SHARD 是 ZeRO-3 风格。每个 block 前向前 AllGather 参数，算完释放完整权重，反向前再 AllGather，反向后 ReduceScatter 梯度。每张卡用归约后的梯度分片和本地优化器状态更新参数分片。

## 面试官可能追问

- **能省掉反向前的 AllGather 吗？** 前向后保留完整权重即可，代价是更高的峰值显存。FSDP1 SHARD_GRAD_OP 采用这个思路，通信量接近 ZeRO-2。
- **反向 AllReduce 优化器状态吗？** 梯度通过 ReduceScatter 归约到负责更新该分片的 rank，Adam 状态由该 rank 本地维护。全局梯度范数等辅助统计可另外归约。

## 关联知识

- [[../../../../wiki/concepts/ZeRO|ZeRO]]
- [[../../../../wiki/concepts/FSDP|FSDP]]

## 参考来源与待核实

- [[../../大模型系统面试题地图#3. ZeRO vs FSDP|大模型系统面试题地图]]

- [PyTorch FSDP1 策略文档](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

## 所属题单

- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
