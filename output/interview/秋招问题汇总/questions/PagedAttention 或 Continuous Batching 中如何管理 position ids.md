# PagedAttention 或 Continuous Batching 中如何管理 position ids？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

position id 属于每条请求的逻辑 token 序列，而非物理 KV page、slot 或当前 batch 行号。调度器为本轮每个新 token 传入其真实逻辑位置；分页映射只决定 KV 存储位置，continuous batching 的请求重排不能改变 RoPE 位置。

## 深入解释

prefix 复用、chunked prefill、滑窗和特殊位置编码要单独核对位置偏移与模型约定。

## 关联知识

- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]
- [[../../../../wiki/concepts/Continuous Batching|Continuous Batching]]

## 参考来源与待核实

- [[../../面试经验#4. 然后往 AI infra 方向深入|面试经验]]


## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
