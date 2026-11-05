# Prefix Caching 如何复用前缀，命中后 attention 执行有何变化？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 历史内容从原稿提炼，核实边界见文末；本次英伟达五面补充单独注明公式假设和依据，不表示历史原稿全部复核或代码已实测。

## 其他问法

- PrefixCache 的作用是什么？和 KVCache 有什么区别？怎么实现？

- Prefix Caching 命中后，attention 的执行与传参有什么变化？
- Prefix cache 命中后，哪些 tensor 复用，哪些仍需重新计算？
- 什么是 Prefix Caching？

## 30 秒回答

KV Cache 是保存的历史 K/V 状态；Prefix Caching 是跨请求或跨调用复用相同前缀状态的机制。命中后跳过重复 Prefill，但新 tokens 的 Attention 仍需读取可见历史。典型实现按块或 radix tree 匹配前缀，配合引用计数、驱逐和必要的写时复制。

## 深入解释

### 英伟达五面补充

- 本次题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 3 题；2026-11-05 收录，实际面试日期与具体轮次未提供。
- 本次原题：PrefixCache 的作用是什么？和 KVCache 有什么区别？怎么实现？

例：两个请求都以“系统提示词＋同一文档”开头、问题不同，可共享公共前缀的 KV。KV Cache 即便不跨请求共享，也能避免同一请求 Decode 时重新投影历史 tokens；二者不是替代关系。

实现要点：

1. 按 token blocks 管理 KV，查找最长可复用前缀。
2. 哈希方案必须包含上文身份，而不只哈希本块 tokens；radix tree 也是可选组织方式。
3. 命中块通过引用计数共享；共享尾块需要修改时按实现进行 copy-on-write。
4. 对可回收缓存执行驱逐，并与缓存感知路由配合。
5. 模型、LoRA、多模态输入、位置语义等影响状态的因素要进入兼容性判断。

vLLM 官方设计描述父块哈希、本块 tokens 与额外标识；该文档的实现只缓存完整 blocks。完全命中时如何取得 logits，以及具体块粒度、哈希函数和参数名，仍要绑定实际部署版本。

常见误区：Prefix hit 减少重复 Prefill，并不意味着 Decode 不再读历史，也不能随便复用出现在中间但上文不同的一段文本 KV。

历史长篇解释沿文末原稿阅读；本次仅补充上述题干对应的机制、公式和边界。

## 关联知识

- [[../../../../wiki/concepts/RadixAttention|RadixAttention]]

- [[../../../../wiki/concepts/KV Cache|KV Cache]]

- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]

## 参考来源与待核实

- [[../../多卡与推理系统面试梳理#7. Prefix Caching 命中后，attention 阶段和不命中时到底差在哪|多卡与推理系统面试梳理]]
- [[../../多卡与推理系统面试梳理#9. 最后给一个高频追问清单|多卡与推理系统面试梳理]]
- [[../../大模型系统面试题地图#A. 已经有较好锚点，可以直接展开|大模型系统面试题地图]]

- 待核实 / 原稿边界：原稿明确这是工程归纳。其“sequence length 变了”主要指本轮新算 query 长度，不应笼统写成历史 KV 长度也缩短；完全命中时如何得到 logits、块粒度复用及 kernel 参数名需具体实现核实。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

### 本次补充依据与边界

- [vLLM Automatic Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
