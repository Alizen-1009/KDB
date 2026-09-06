# PagedAttention 如何共享前缀物理块，并通过 Copy-on-Write 支持 Beam Search 分叉？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- 多个序列共享相同 Prompt 前缀时，PagedAttention 如何复用物理块？
- PagedAttention 的 Copy-on-Write 原理是什么，解决什么问题？
- 结合 Beam Search 说明 PagedAttention 的内存共享。

## 30 秒回答

各序列保留独立 block table，但相同前缀的逻辑块可以指向同一物理块，并用引用计数跟踪共享。只读时不复制；某序列要修改仍被共享的块时，先复制内容到新块，再只更新自己的映射。Beam Search 因而能共享公共历史，在后缀分叉时形成独立路径。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]
- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]

## 参考来源与待核实

- [[../../推理系统专题面试稿#3. 在 PagedAttention 中，当多个序列共享相同的 Prompt 前缀时，物理块是如何复用的？请说明 Copy-on-Write 机制的工作原理。|推理系统专题面试稿]]
- [[../../推理系统专题面试稿#4. 在 PagedAttention 的 Block 管理中，如何实现多个序列之间的内存共享？请结合 Beam Search 场景说明。|推理系统专题面试稿]]
- [[../../推理系统专题面试稿#5. 请解释 PagedAttention 中 Copy-on-Write（写时复制）机制的原理，以及它在 LLM 推理中解决了什么问题？|推理系统专题面试稿]]

- 待核实 / 原稿边界：同一文件第 3、4、5 题均围绕 block table、引用计数与 CoW，已合并。CoW 发生在写入共享块时，新块分配不等同于总要复制整段 KV；具体框架版本的 beam/prefix 支持仍需核实。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
