# 千问 C 端 AI Infra 一面

这组问题来自用户提供的千问 C 端 AI Infra 一面复盘，顺序按面试回忆保留，不代表考频统计。只有题干的新题先标记为“待整理”，不为整批题补造答案。

## 题目

1. 个人题：常规过简历。应结合真实项目、个人职责和结果回答，未经授权不在 KDB 编写个人经历答案。
2. [[../questions/PD 解耦和 PD 分离有什么区别|介绍 PD 分离。]]
3. [[../questions/PD 分离中的 Prefill 与 Decode 实例配比如何确定|PD 分离中的 Prefill 与 Decode 实例配比如何确定？]]
4. [[../questions/SGLang RadixCache 如何组织前缀复用、引用与淘汰|SGLang RadixCache 的原理是什么？]]
5. 个人题：有没有完整阅读过 SGLang 源码，有没有尝试改进过？需要以实际读过的模块、commit、修改和验证结果回答。
6. [[../questions/KV Cache 如何工作，为什么既加速推理又带来访存瓶颈|详细介绍 KV Cache 的原理。]]
7. [[../questions/SGLang HiCache 如何实现分层 KV Cache|HiCache 是什么？]]
8. [[../questions/PD 分离中的 Router 如何进行负载与缓存感知路由|PD 分离中还有哪些先进的 Router 实现？]]
9. 个人题：有没有用 CUDA 写过新算法的算子？需要结合真实算法、正确性验证和性能数据回答。
10. [[../questions/CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|介绍一下 CUTLASS。]]
11. [[../questions/FlashAttention-1、2、3 分别优化了什么|详细介绍 FlashAttention-3 的改进点。]]
12. [[../questions/国产 AI 加速卡与 NVIDIA GPU 的软件栈和优化差异是什么|用过的国产硬件与 NVIDIA GPU 主要有什么区别？]]
13. [[../questions/如何用 Roofline 模型区分计算、访存和其他瓶颈|介绍 Roofline 模型。]]
14. [[../questions/如何用 Roofline 模型区分计算、访存和其他瓶颈|调优 kernel 时，如何通过 Roofline 定位并改进瓶颈？]]
15. [[../questions/Online Softmax 为什么可以分块计算，再合并成完整 softmax|Online Softmax 与普通 Softmax 的计算量如何比较？]]
16. [[../questions/如何区分计算密集型与访存密集型算子|介绍计算密集型与访存密集型算子。]]
17. [[../questions/Linear Attention 为什么不能全面替代 softmax attention|介绍线性注意力。]]
18. [[../questions/PyTorch 算子如何注册 CPU 与 CUDA 实现并由 Dispatcher 分发|Torch 算子如何分别注册 CPU、CUDA 实现，Dispatcher 如何分发？]]
19. [[../questions/XLA 与 TVM 等 AI 编译栈如何优化计算图，有何侧重|简单介绍图优化。]]
20. [[../questions/如何完整实现带 Reduce 的 CUDA RMSNorm|Coding：完整实现 CUDA RMSNorm，包括 Reduce。]]
21. [[../questions/如何用 Triton 实现并优化 Softmax|Coding：实现 Triton Softmax。]]

## 返回

- [[../README|秋招问题汇总]]
