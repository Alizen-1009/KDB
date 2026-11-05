# Context Parallel 中 AllGather KV 与 AllGather Q 如何选择？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 7 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

CP AG KV 和 AG Q 方案怎么选择？Agent 场景还能 AG KV 吗？

## 30 秒回答

AG KV 收集完整 KV、分别计算本地 Q；AG Q 则让持有各 KV shard 的 rank 获得 Q，计算局部 Attention 后合并输出与 Softmax 统计。选择看实际张量字节、内存和同步开销。长历史、短新增 Q 的 Agent 通常更适合 KV 留在原地，但不是绝对禁止 AG KV。

## 深入解释

AG 在本页解释为 AllGather；原图未指定框架、版本和布局，因此只讲通用数据流，不声称某版本默认采用哪种算法。

| 路线 | 计算布局 | 必须考虑的通信 |
|---|---|---|
| AG KV | 各 rank 收集 KV，计算本地 Q 对应输出 | KV 聚合；完整 KV 的临时容量 |
| AG Q | 各 rank 收集所需 Q，以本地 KV 算 partial Attention | Q，以及 partial output 和 LSE 的合并 |

普通注意力下，可先比较未乘 collective 拓扑系数的张量大小：

$$Q_{bytes}\sim BL_qH_qd_qb$$
$$KV_{bytes}\sim BL_{kv}H_{kv}(d_k+d_v)b$$

还需计算 partial output 的形状、collective 次数和工作区。MLA 必须用 latent/位置分量的真实布局代替普通 KV-head 公式。

AG Q 路径的局部归一化输出 Oᵣ 不能直接相加，需用局部 LSE：

$$L=\operatorname{logsumexp}_r L_r,\qquad O=\sum_r e^{L_r-L}O_r$$

Agent 常见 L_q≪L_kv，因此保留历史分片、交换 Q 与小结果更有吸引力。但历史 KV 已常驻且只同步新增部分、网络/算子支持不同，或 workload 有大 Q 时，AG KV 仍可能合理。不要混淆“每轮全量 AG 历史”和“只交换增量后复用”。

## 关联知识

- [[../../../../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]
- [[../../../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]
- [[../../../../wiki/concepts/MLA|MLA]]

## 参考来源与待核实

- 通用通信量与 Softmax 合并为机制推导；具体框架支持、collective 和 layout 待绑定版本确认。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
