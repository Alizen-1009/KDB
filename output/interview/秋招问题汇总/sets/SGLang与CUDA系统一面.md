# SGLang 与 CUDA 系统一面

这组问题来自用户提供的一次一面复盘，主线是从 SGLang 项目一路下钻到 RadixCache、分页 KV、varlen kernel metadata、显存 slot，再扩展到 PD 分离、投机解码、量化和 CUDA 工具栈。顺序按面试回忆保留，不代表考频统计。

## 题目

1. 个人项目：挑选简历中的 SGLang 项目深入追问职责、难点和结果。此项涉及个人经历，暂不写入 KDB 通用答案；若要整理建议另行授权写入 `my_resume`。
2. [[../questions/vLLM、TGI、TensorRT-LLM、SGLang 如何比较|如何介绍 SGLang，它与其他推理引擎的侧重点有何不同？]]
3. [[../questions/SGLang RadixCache 如何组织前缀复用、引用与淘汰|SGLang RadixCache 如何组织前缀复用、引用与淘汰？]]
4. [[../questions/LLM 推理显存由哪些部分组成，如何按对象优化|SGLang 推理显存由哪些部分组成？]]
5. [[../questions/SGLang 显存不足时如何回收 KV Cache，哪些显存不能被 RadixCache 淘汰|SGLang 显存不足时如何处理，有没有看过相关源码？]]
6. [[../questions/PagedAttention 的原理是什么，解决了什么痛点|分页 KV 显存管理如何工作？]]
7. [[../questions/SGLang 的 varlen metadata 如何映射到物理 KV slot|调用 varlen kernel 时 metadata 如何组织，逻辑 token 如何对应物理 slot？]]
8. [[../questions/一个 KV page 内的多个 slot 在显存中是否连续|一个 page 内的多个 slot 在显存中是否连续？]]
9. [[../questions/PD 解耦和 PD 分离有什么区别|PD 解耦和 PD 分离有什么区别？]]
10. [[../questions/PD 解耦与物理分离如何实现，KV 和请求状态怎样交接|PD 分离如何交接 KV 和请求状态？]]
11. [[../questions/PD 分离中的 Prefill 与 Decode 实例配比如何确定|PD 分离中的 Prefill 与 Decode 配比如何确定？]]
12. [[../questions/MTP 与 DSpark 的基本思路和区别是什么|MTP 与 DSpark 的基本思路和区别是什么？]]
13. [[../questions/FP8、INT8、AWQ、GPTQ、SmoothQuant 有何区别，各适用于什么场景|主流量化方法如何分类，SmoothQuant 的原理是什么？]]
14. [[../questions/CUDA 语境中的全双工、半双工与 cache line 分别是什么|CUDA cache line 与全双工、半双工分别是什么意思？]]
15. [[../questions/如何用 Roofline 模型区分计算、访存和其他瓶颈|Roofline 中的指标如何理解？]]
16. [[../questions/CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|Thrust、CUB 与 CUTLASS 各自解决什么问题？]]

## 本轮重点

- [[../questions/SGLang 的 varlen metadata 如何映射到物理 KV slot|varlen metadata 与物理 slot]]
- [[../questions/一个 KV page 内的多个 slot 在显存中是否连续|page 内 slot 连续性]]
- [[../questions/CUDA 语境中的全双工、半双工与 cache line 分别是什么|全双工、半双工与 cache line]]
- [[../questions/CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|Thrust、CUB 与 CUTLASS]]

## 返回

- [[../README|秋招问题汇总]]
