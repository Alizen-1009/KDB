---
type: source
source_kind: 文章
topic: 投机解码
updated:
---

# 并行投机解码(DFlash/DSpark)的快速理解与vLLM实测

## 来源信息

- 标题：并行投机解码(DFlash/DSpark)的快速理解与vLLM实测
- 作者：kaiyuan（kaiyuan InfraTech）
- 日期：2026-08-17
- 类型：文章
- 原始文件：[[../../raw/articles/并行投机解码(DFlashDSpark)的快速理解与vLLM实测|并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]
- 原始链接：[微信公众号](https://mp.weixin.qq.com/s/9H8_PDcwMvMznOtJQ9ENZg)

## 2-3 条核心摘要

- 文章把 [[并行投机解码]] 与传统自回归 drafter 区分开：[[DFlash]] 用 diffusion 式 masked filling 一次并行预测多个 draft token，再由 target model 做常规 speculative verification，目标是减少 drafter 自回归串行链。
- DFlash 融合 target model 若干中间层 hidden states，并把投影后的 target context 拼入 drafter 每层 Attention 的 K/V，以提高并行草稿质量；[[DSpark]] 进一步加入顺序生成与 confidence 估计，再由 Hardware-Aware Prefix Scheduler 根据候选前缀存活概率和硬件 `SPS(B)` 曲线决定实际验证长度。
- 来源在 Qwen3-4B、A800 与其所称 vLLM `0.26.0` 环境中观察到：DSpark `561–584 tok/s`、DFlash `449–480 tok/s`、Baseline `229 tok/s`。这些是特定实验的来源结果；GPU 实际参与数、TP 配置、镜像版本和完整 workload 仍不够明确。

## 值得关注的论断

- DSpark 的主要系统创新不是简单增加固定 draft length，而是把“验证多少候选”改成随候选 confidence 和硬件负载变化的在线优化问题：草稿可以较长，但低价值后缀不必进入 target verification。
- 来源称两组 `num_speculative_tokens=4/7` 交叉配置下，DSpark 相对 DFlash 的吞吐优势约为 `1.17x–1.30x`，相对 Baseline 约为 `2.45x–2.55x`；DFlash 相对 Baseline 约为 `1.96x–2.09x`。
- 文章把 temperature=0 时同一 Baseline 两次 GSM8K 评测相差 `5.2pp` 归因于 vLLM 不同 batch 组合改变浮点累加顺序，进而让边界 token 的 argmax 发生变化。该解释与 [[确定性推理]] 的机制方向一致，但本文没有提供逐 token 数值追踪，应保留为来源归因。

## 机制拆解

### DFlash

1. Target model 对当前 prefix forward，并预测一个 anchor token。
2. 抽取 target 若干中间层 hidden states，融合为上下文 `H_ctx`。
3. Drafter 以 anchor 和 mask positions 为 token 侧输入；每层 Attention 的 Query 来自 draft hidden `H_d`，K/V 则读取投影后的 `[H_ctx || H_d]`。
4. Drafter 对多个 mask 位置并行产生候选 token，最后仍由 target model 统一验证。

### DSpark

1. Parallel block 根据 anchor、mask 与 target context 产生多个位置的中间 logits。
2. Sequential block 从左到右生成候选 token，并输出条件 confidence `c_k`。
3. 第 `j` 位的前缀存活概率为 `a_j = ∏_{i≤j} c_i`。
4. 系统预先 profile `SPS(B)`：验证 batch 含 `B` 个 token 时，每秒可执行多少 target forward steps。
5. 对各请求验证长度 `l_r`，文章给出 `B = Σ_r(1 + l_r)`，并以期望接受量 `τ` 和 `Θ = τ · SPS(B)` 估计吞吐。
6. Scheduler 按候选扩展的存活概率贪心加入更多验证 token；当 `Θ` 不再提升时停止，只把选中的前缀送给 target，后缀直接丢弃。

## 来源 Benchmark

### 环境与配置

- 硬件：文章写作 `8× NVIDIA A800-SXM4-80GB`。
- Target：Qwen3-4B。
- Drafters：DSpark `block7`、DFlash `b16`。
- 框架：文章写作 vLLM `0.26.0`，启动镜像为 `vllm/vllm-openai:latest`。
- 主配置：DSpark `num_speculative_tokens=4`，DFlash `=7`；另做 `7/4` 参数交叉消融。

### 主要结果

| 配置 | GSM8K | MMLU | 吞吐量 | vs Baseline |
| --- | ---: | ---: | ---: | ---: |
| DSpark，num_spec=7 | 35.6% | 27.6% | 584 tok/s | 2.55x |
| DSpark，num_spec=4 | 35.2% | 28.8% | 561 tok/s | 2.45x |
| DFlash，num_spec=4 | 32.0% | 27.2% | 449 tok/s | 1.96x |
| DFlash，num_spec=7 | 31.6% | 27.6% | 480 tok/s | 2.09x |
| Baseline | 29.2% / 34.4% | 28.0% / 29.2% | 229 tok/s | 1.0x |

## Benchmark 边界

- 文章称使用 8 张 A800，但每个容器只暴露 4 张 GPU，启动命令没有显式 `--tensor-parallel-size`；Qwen3-4B 可单卡运行，因此实际计算 GPU 数与 TP 配置待核实。
- `latest` 镜像标签不可复现；需要保存 image digest、vLLM commit 和模型 revision。
- 正文没有完整记录并发、请求数、输入/输出长度分布、warmup、采样参数和多次 trial 方差，吞吐数字不能外推到其他 serving workload。
- GSM8K/MMLU 各 250 题，`1pp` 左右差异的统计显著性有限；“num_speculative_tokens 基本不影响精度”只是在 `4/7` 两个值上的当前观察。
- 正确实现的 speculative verification 应维持 target model 的生成规则；任务准确率差异不应直接解释为 drafter 改善模型质量，应先排查 batch-dependent 数值路径、采样配置和评测噪声。

## 关键概念

- [[并行投机解码]]
- [[DFlash]]
- [[DSpark]]
- [[Speculative Decoding]]
- [[Benchmarking]]
- [[确定性推理]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/kaiyuan]]

## 与现有 wiki 的关系

- 更新 [[Speculative Decoding]]：补充 diffusion-style parallel drafter 和 adaptive verification length。
- 更新 [[Benchmarking]]：补充 inference benchmark 的版本、并行拓扑、负载与数值波动控制。
- 更新 [[确定性推理]]：补充 temperature=0 下 batch 形态仍可能影响输出的来源案例。
- 无直接冲突；它扩展了现有 wiki 以 sequential drafter / MTP 为主的投机解码路线。

## 待确认

- DFlash target hidden fusion 的具体层选择、融合函数、K/V 投影和训练目标应以官方 repo / 论文核对。
- DSpark sequential block、confidence calibration、scheduler 复杂度和 `SPS(B)` profiling 更新策略应以原论文 `arXiv:2607.05147` 核对。
- vLLM 对 `method=dflash/dspark` 的正式支持版本、CLI、并行限制和代码路径需绑定源码 commit 验证。
- 复现实验需补 image digest、TP/DP 配置、实际 GPU utilization、数据集版本和完整 benchmark 命令。
