---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-11-09
---

# AI Infra 早报｜vLLM 让 MoE 下投影少搬一次数据

## 来源信息

- 标题：AI Infra 早报｜vLLM 让 MoE 下投影少搬一次数据
- 作者：Ethan / Miracle Farms
- 日期：2026-11-05 08:37（正文署时）；剪藏日期 2026-11-09
- 类型：文章 / 技术早报（二手整理）
- 原始文件：[[../../raw/articles/AI Infra 早报｜vLLM 让 MoE 下投影少搬一次数据]]
- 原文：[微信文章](https://mp.weixin.qq.com/s/MxjkwKiI0M3n2FCcoWoJWQ)
- 阅读范围：完整剪藏正文；后于 2026-11-09 补读 vLLM PR #64760 的说明、diff 与测试命令。FlashInfer release、LangGraph PR 未独立核验，未复测 GPU 性能；远程题图不作为实现证据。

## 2-3 条核心摘要

- **布局衔接**：文章称 vLLM PR #64760 在 GLM 5.3、H200 的 FP8 MoE 下投影路径中保持 expert-sorted 中间数据，使激活与专家权重适合通过 TMA 搬运，减少阶段间布局整理。
- **激活与量化融合**：将 SiLU、逐元素乘法和 FP8 量化收进更紧凑的执行路径，目标是减少中间张量读写和 kernel launch；PR 核验确认使用独立的激活量化 Triton kernel，并非融合进 GEMM epilogue。
- **收益与适用边界**：来源转述端到端吞吐提升 6.0%–7.6%，剪藏未列完整条件，后续核对 PR 已补齐 TP8、输入 2048、输出 1、请求并发 1 等条件，详见下节；仍记录为 PR 测试声称，不能泛化到所有 MoE、GPU 或后端。

## 值得关注的论断

| 来源说法 | 证据边界与整理 |
| --- | --- |
| 同一专家的数据保持在一起，下投影同时通过 TMA 读取激活和权重 | PR diff 已确认 GEMM1 sorted store 与 GEMM2 TMA 连续块加载；完整适用条件见下节。 |
| 下投影“少搬一次数据” | 标题概括，不是已核实的物理复制次数；需要对照优化前后张量落地、重排及 GEMM loader 路径。 |
| 端到端提升 6.0%–7.6% | 来源声称，未复测；不能据此单独归因于 TMA、布局或融合。 |
| vLLM PR 已合并，但稳定版收录未证实 | 2026-11-09 已通过 PR 元数据确认 merged；首个包含 release 仍待核实。 |
| 更专用的 kernel 需要承担迁移与维护成本 | 作者的工程判断；应结合 shape、量化格式和 GPU 后端验证。 |

### PR 一手核验补充（2026-11-09）

依据：[PR #64760](https://github.com/vllm-project/vllm/pull/64760)，head `d5c86e9a065717f84e62af57903b9c5121725923`，merge commit `42c043c6b0e4905d590e9cb4bf00bb644bfebb53`。此节区分于上述二手文章转述，不将对话当作新事实来源。

**范围**：优化 GPU 内专家 FFN 的两次 GEMM 衔接，不涉及减少 EP All-to-All。示例部署命令为 `-tp 8`，未显式启用 EP；快路径条件还要求 `expert_map is None`。因此不能由这份 PR 推导任意 EP backend 都能使用同一优化。

| 阶段 | 原路径 | 新快路径 |
| --- | --- | --- |
| GEMM1：gate/up | 按专家调度计算，输出写到原 route 行 | `C_is_sorted=True`，直接写到 expert-sorted 物理行 |
| 激活与量化 | 不据摘要假定所有旧分支都分开执行 | 独立 `_silu_mul_quant_sorted_kernel` 完成 SiLU、乘法、FP8 量化和 scale 写出，保持排序 |
| GEMM2：down | 按路由映射从激活取行 | `A_is_sorted=True`，使用 `a_desc.load` 连续分块加载激活，配合权重 TMA |

LLM 归纳：可理解为推迟恢复 route 顺序，避免在两次 GEMM 之间先散写再按专家间接读。这里可能是 GEMM store/loader 内的索引变化，不代表必然删掉一个独立 copy kernel；高精度 GEMM1 中间结果和 FP8 激活仍会落地。下投影属于专家 FFN 内部的第二次矩阵乘，并非“专家计算结束后的外部阶段”。

**适用约束**：代码检查 FP8 W8A8、block shape `[128,128]`、SiLU 无 clamp、无 LoRA/相关 hook、非量化模拟、动态 a2 scale、E8M0 路径、CUDA SM90、连续的 w2 及 shape/tile 整除条件，并需调优配置启用 `USE_TMA`。不能仅凭 H200、MoE 或 TP8 就认定一定触发。

**已给出的 benchmark 条件**：H200 / `zai-org/GLM-5.3`，TP8、prefix caching、`max-num-seqs=48`、MTP 草稿数 5；随机输入 2048、输出 1，32 个请求、4 次 warmup、请求并发上限 1。PR 声称的 6.0%–7.6% 为此类 Prefill 主导场景的端到端收益，不是 decode TPOT 提升。`max-num-seqs=48` 是服务容量参数，不是该次实测请求并发。具体 checkpoint revision、软件完整构建环境与独立复测仍缺失。

**正确性证据**：PR 加入 H200 单测对 fast path 与 route-major 路径使用 `torch.equal`，并列出 AIME25 结果；本次仅检查测试代码及报告，没有本地运行，不能外推所有 shape 与模型均已验证。

**对初次摘要的更正**：先前“输入输出长度与并行条件未说明”仅适用于剪藏文章，不能继续描述原 PR；“融合是否进入 epilogue”现已核实为独立 kernel；合并状态已确认，稳定版收录仍未确认。

### 旁支消息

- FlashInfer：文章称 v0.7.3 已发布，为 Rubin 增加 MTP 与 NVFP4 MLA decode，为 Blackwell 优化 EP MoE decode，并为 Hopper 增加 MXFP4/NVFP4 experts。均按文章发布时的 release 转述记录；kernel 路径、模型支持、精度与性能矩阵待核实，不能由支持某存储格式推导硬件原生计算能力。
- LangGraph：文章称 Python SDK PR #9568 已合并，允许选择 thread state/history 的字段路径，如 `values.messages`；SDK 首个版本、服务端兼容性及真实负载收益待核实。仅在本来源保留，不单建实体。

### 推荐验证维度（LLM 归纳）

- 数据流：gate/up GEMM → SiLU×up → 量化 → down GEMM → combine；逐项记录布局、dtype、scale 生成和中间存储。
- 性能：端到端吞吐、TTFT/TPOT 与尾延迟，结合激活/量化/下投影 kernel 时间、HBM 流量、launch 数及显存占用判断；少一个重排不等于整体带宽瓶颈消失。
- 覆盖：固定 commit、GPU、模型 checkpoint、FP8 策略、并行方式、batch、序列长度与基线，确认目标后端实际被选中。

## 关键概念

- [[../concepts/MoE]]
- [[../concepts/算子融合]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/FlashInfer]]

## 与现有 wiki 的关系

- 局部更新 MoE 与算子融合的阶段间布局、激活量化融合案例；不另建重复的融合概念。
- 更新 vLLM、FlashInfer 的版本绑定线索与来源回链。
- 既有 MoE 页的 C1 epilogue 生成 C2 输入是另一类融合边界；本 PR 使用独立激活量化 kernel，两者不矛盾。已显式更正初次摘要对 PR 测试条件、融合边界和合并状态的待核实表述。
- 既有 vLLM 页的 GLM 支持矩阵绑定旧 commit；本篇的后续 PR 描述不能倒推旧 commit 原生支持同一模型，也不能把“GLM 5.3”自动等同于“GLM-5.3-Flash”。

## 待确认

- [vLLM PR #64760](https://github.com/vllm-project/vllm/pull/64760)：首个包含 release、目标 EP/后端是否能触发、完整环境与 checkpoint revision、各 shape 独立性能及精度复测；diff、合并 commit 和已披露测试条件见上节。
- [FlashInfer v0.7.3](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.3)：文章列出的功能与硬件矩阵、dispatch 条件和实测收益。
- [LangGraph PR #9568](https://github.com/langchain-ai/langgraph/pull/9568)：SDK/service 兼容关系与字段选择的序列化、传输和执行收益。
