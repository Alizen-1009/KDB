# 有 KV Cache 时，RoPE 应在写缓存前做还是读缓存时做？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

常见实现是在 K 写入 KV Cache 前按 token 的逻辑位置完成 RoPE；之后每步 decode 可直接读取已旋转 K，避免重复旋转历史。当前 Q 每步按当前位置旋转，V 通常保持原样。

## 深入解释

也可设计缓存未旋转 K 的方案，但读时要知道每个历史位置并承担重复计算；修改 theta/scaling 后旧 cache 不能直接按新配置解释。

## 关联知识

- [[../../../../wiki/concepts/KV Cache|KV Cache]]
- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
- [[../../面试经验#4. 然后往 AI infra 方向深入|面试经验]]


## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
