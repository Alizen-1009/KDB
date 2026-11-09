---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-11-09
---

# SGLang 低延迟模式：Kernel 与运行时

## 来源信息

- 标题：SGLang Low-Latency Mode: Kernels and Runtime（中文翻译）
- 作者：SGLang、NVIDIA 与 AMD 团队
- 日期：原文标注 2026-10-18；译文标注 2026-10-20
- 类型：文章
- 原始文件：[[../../raw/articles/SGLang 低延迟模式：Kernel 与运行时.md]]
- 原文：https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode

## 2-3 条核心摘要

- 面向低并发、小 batch decode 与投机验证，低延迟模式用外部插件接入模型和硬件专用 kernel，主要通过缩短依赖等待、通信和执行边界改善关键路径。
- NVIDIA 在多 kernel 路径中提前释放 PDL、在投影 GEMM epilogue 发起跨 GPU 发送，并优化小 batch MoE 和 DSpark 采样；AMD 将每层大部分工作组织成 K1/K2 两次 persistent 启动，在 kernel 内重叠加载、计算和规约。
- 插件声明支持配置，初始化时 guard、构建并计入显存；运行时按步骤类型和 token 数分发，在支持的服务配置中未覆盖的步骤回退通用前向。Graph 捕获记录选定路径，多 TP rank 须一致。

## 值得关注的论断

- 4×GB300、DeepSeek-V4.1-Flash + DSpark、单请求验证 profile 中，graph kernel 数从 1,205 降到 1,189，decoder 层中位数从 96.8 µs 降到 68.4 µs；不同区域中位数不能相加，profiling 会影响计时，不等于 serving 性能。
- AgentX 回放中，相对同硬件 SGLang 基线，4×GB300 单会话 P90 单用户生成速度 671.2→928.4 tokens/s（1.38×）；4×MI355X、4 会话 331.6→516.0 tokens/s（1.56×）。这是来源报告的特定模型与负载结果，不可直接外推。
- 另一个 4×GB300、8K 输入/1K 输出、1–8 并发的 serving 测试报告单用户 decode 速度提升 12–38%；两组仅切换插件，模拟平均接受长度均为 3.51。它不是上述 AgentX 指标。
- NVIDIA 在 GEMM epilogue 推送 tile，但前后仍是两个 kernel；AMD 在 K1/K2 边界保证 KV 写入对 GPU 可见，并为跨 GPU 等待设置超时。两种路线都不是消除所有同步。

## 关键概念

- [[../concepts/Programmatic Dependent Launch]]
- [[../concepts/Persistent Kernel]]
- [[../concepts/Megakernel]]
- [[../concepts/通信-计算重叠]]
- [[../concepts/CUDA Graph 执行模式]]
- [[../concepts/DSpark]]

## 相关实体

- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 补充 PDL、persistent kernel、通信重叠、graph 和 DSpark 的低延迟实现实例；AMD 按层两次启动，并非整模型单 kernel。
- 未发现与已读页面的直接冲突；专用插件不意味着所有配置都支持或总能加速。

## 待确认

- 插件接口、受支持硬件/模型/并行配置、回退及 CI 覆盖需按实际 SGLang 与外部库版本核实；文章称接口仍在演进，部分 CI 为未来计划。
- 性能与正确性结论来自文章，尚未在本地复现；图表为远程图片，本摘要只引用文章正文明确给出的数值。
