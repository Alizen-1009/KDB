---
type: source
source_kind: 文章
topic: KV Cache
updated: 2026-11-09
---

# KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？

## 来源信息

- 标题：KV Cache 研习社 · 深潜 | LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？
- 作者：KV Cache 研习社 / Q仔学AI
- 日期：原文标注 2026-09-14
- 类型：技术评论 / 文章
- 原始文件：[[../../raw/articles/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？.md]]
- 原文链接：https://mp.weixin.qq.com/s/utA9JDybd7nYVkoj2T-2JA

## 2-3 条核心摘要

- 文章将 [[../entities/LMCache]]、[[../entities/Mooncake]]、[[../entities/FlexKV]] 分别归纳为独立缓存层、集群级 KV 池与多级卸载管道；三者能力重叠，并非互斥分类。
- LMCache MP 模式据文章描述在多个推理副本间共享主机内存缓存；Mooncake 结合 PD 分离和跨节点传输；FlexKV 侧重多级存储与异步搬运。
- 跨请求复用可以避免重复 prefill，但缓存是否值得读取须比较传输、排队与重算成本；平台实现和模型 KV 布局兼容性同样重要。

## 值得关注的论断

- 文章称 LMCache MP 在 Qwen3-235B-A22B-Instruct-2507-FP8、8×H100、vLLM 0.18.1、LMCache 0.4.3-dev、多轮 QPS 2、120 秒条件下，平均 TTFT 从 3.98 秒降到 0.29 秒；具体对照及复现待查原始基准。
- 文章转述 Mooncake 在真实负载下请求容量提升 75%，在模拟极端场景下吞吐最高提升 525%；不能将后者表述为生产实测或跨工作负载通用结论。
- 文章称 FlexKV 在腾讯云生产环境将并发用户数提升 3–10 倍、对比纯 SSD 卸载延迟降低 67%；实验配置与一手证据待核实。
- 文章声称 LMCache、FlexKV 分别可与 Mooncake Transfer Engine 集成；三者联合堆叠是文章设想，不等于已验证存在可直接部署的组合。

## 关键概念

- [[../concepts/分层 KV Cache]]
- [[../concepts/Prefix Caching]]
- [[../concepts/PD分离]]

## 相关实体

- [[../entities/LMCache]]
- [[../entities/Mooncake]]
- [[../entities/FlexKV]]

## 与现有 wiki 的关系

- 补充分层存储、前缀复用及 PD 交接的项目案例，与既有权衡不冲突。
- 原文将三项目定位为层、池、管道是解释性分类，不代表各自能力范围完全分离。

## 待确认

- 官方版本、实际集成 PR、生产案例、基准条件和缓存数据布局兼容性。
- RDMA 直达 GPU 依赖硬件和软件路径，不能把零拷贝写成所有配置的保证；自动分层与异步搬运并非没有开销。
- 原文插图以远程图片引用，正文未完整展开图中细节；未独立核验图和其引用文献。
- “唯一引擎无关”“吞吐最优”“最完整”等比较性判断缺少可比证据；原文将线性注意力一概描述为消除增长也不适用于所有混合架构。
