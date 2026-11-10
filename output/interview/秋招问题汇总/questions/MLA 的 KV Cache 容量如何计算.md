# MLA 的 KV Cache 容量如何计算？

- 整理状态：已整理
- 题目出处：用户本次提供

## 30 秒回答

对缓存共享 latent 和独立 RoPE key 的典型 absorbed MLA，设 latent 维 C、RoPE key 维 R，每层有效载荷为 `B×S×(C+R)×元素字节数`，同构 L 层再乘 L；不应套标准 K/V 两份、KV head 数的公式，也不应漏掉 RoPE key。题目未给 C、R、层数和实际 dtype，不能算唯一数值。物理布局、量化 scale、分页、TP 复制或分片和额外 index cache 都可能改变实际占用，不同 MLA 模型也可能采用其他结构。

依据：[[../../../../wiki/concepts/MLA|MLA]]、[[../../../../wiki/concepts/KV Cache|KV Cache]]；对比 [[如何用最小代码解释 MLA 的 latent KV 结构及缓存收益|简化 MLA 示例]]。

## 所属题单

- [[../sets/长上下文与全局缓存调度面试题|长上下文与全局缓存调度面试题]]
