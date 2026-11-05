# MQA 和 GQA 为什么比 MHA 更节省 KV Cache？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 历史内容从原稿提炼，核实边界见文末；本次英伟达五面补充单独注明公式假设和依据，不表示历史原稿全部复核或代码已实测。

## 其他问法

- Attention 变化：MHA、MQA、GQA、MLA，对算子的影响是什么？计算访存比如何变化？

## 30 秒回答

在相同 Q heads、序列长度和 head dim 下，GQA/MQA 减少 KV heads，降低缓存与读取量，但 QK/PV 计算不按相同比例下降。理想 Decode Attention 算术强度随 H_q/H_kv 增加。MLA 存 latent KV，需要按展开或吸收模式重算 FLOPs/bytes，不能直接套普通 KV-head 公式。

## 深入解释

### 英伟达五面补充

- 本次题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 硬件与算子第 1 题；2026-11-05 收录，实际面试日期与具体轮次未提供。
- 本次原题：Attention 变化：MHA、MQA、GQA、MLA，对算子的影响是什么？计算访存比如何变化？

假设 Decode 每请求一个 Q，Q/K/V 的 head dim 都为 d，batch 为 B，历史长 S，每元素 b bytes。忽略投影、Softmax、分页和输出流量：

$$FLOPs\approx4BSH_qd$$
$$KV_{bytes}\approx2BSH_{kv}db$$
$$I\approx\frac{2H_q}{bH_{kv}}$$

BF16 下 b=2，故 I≈H_q/H_kv FLOP/byte。这个理想公式要求同一份 KV 真正被多个 Q heads 复用，若 kernel 重复读取则收益变小。

| 结构 | KV 组织 | 算子影响 |
|---|---|---|
| MHA | H_kv=H_q | KV 大，通常更偏带宽受限 |
| GQA | 多个 Q heads 共享一组 KV | 减少 KV heads、投影与读流量；核心 QK/PV 不同比例下降 |
| MQA | 全部 Q heads 共享一组 KV | KV 更小，要求 kernel 高效跨头复用 |
| MLA | latent KV 加独立位置部分 | 上投影、矩阵吸收、latent 聚合改变数据流 |

MLA 展开模式与吸收模式的 GEMM 维度和 FLOPs 不同，必须先确定执行路径。Prefill 的长 Q 本就提供更多复用，且线性投影也占成本，不能把 Decode 强度公式直接当作 Prefill 或整个模型的加速倍数。

历史长篇解释沿文末原稿阅读；本次仅补充上述题干对应的机制、公式和边界。

## 关联知识

- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]

- [[../../../../wiki/concepts/MLA|MLA]]

- [[../../../../wiki/concepts/KV Cache|KV Cache]]

## 参考来源与待核实

- [[../../面试经验#4. 模型基础架构|面试经验]]
- [[../../面试经验#5. KV cache 的缓存和计算|面试经验]]

- 待核实 / 原稿边界：历史原稿仅作容量归纳；本次补充展开头共享结构和 Decode 核心算术强度，不把缓存下降等同于总 Attention FLOPs 同比例下降。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

### 本次补充依据与边界

- 以上为理想 Decode 核心 Attention 机制推导，非完整层/模型 benchmark；MLA 路径参见 [DeepSeek 官方示例](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
