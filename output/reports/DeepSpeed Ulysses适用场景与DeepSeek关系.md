# DeepSpeed Ulysses 适用场景与 DeepSeek 的关系

## 先纠正名称

`DeepSpeed Ulysses` 不是 DeepSeek 提出的技术，而是 **Microsoft DeepSpeed 团队**面向极长序列 Transformer 训练提出的 sequence parallel 方法。它与 DeepSeek 没有“必须优先服务自家模型”的关系；两者只是名称相近。

相关页面：[[../../wiki/concepts/DeepSpeed Ulysses|DeepSpeed Ulysses]]、[[../../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]、[[../../wiki/concepts/Ring Attention|Ring Attention]]、[[../../wiki/concepts/MLA|MLA]]。

## 它解决什么问题

训练超长序列时，即使模型参数能通过 DP/ZeRO/TP 放下，单个 sequence 的 Attention 计算与激活仍可能超过单卡能力。Ulysses 让每张卡先持有 `S/P` token：

```text
输入布局：每 rank [S/P, H]
```

Attention 前对 Q/K/V 做 All-to-All：

```text
[S/P, H] → [S, H/P]
```

每张卡现在拥有完整 sequence、部分 heads，可调用本地 SDPA/FlashAttention。Attention 后再执行反向 All-to-All：

```text
[S, H/P] → [S/P, H]
```

这样 Norm、Residual、MLP、loss 等仍可保持 sequence-sharded；训练反向传播也沿相同布局工作。

## 最适合的场景

- 训练或 SFT，而不是 autoregressive Decode serving；
- sequence 很长，Attention FLOPs 和 activation memory 是主要瓶颈；
- Q heads 足够多，且 `num_q_heads % SP == 0`；
- 节点内 NVLink/NVSwitch 或跨节点高速网络能高效支持 All-to-All；
- 希望复用成熟本地 Attention kernel，不想实现 Ring Attention 的多轮 P2P 与 distributed online softmax；
- GPT/MHA、Q heads 较多的 GQA LLM、长文本训练，以及长视频/ViT token 训练。

DeepSpeed 官方教程定位就是“training Transformer models with extreme long sequences”，当前代码还提供 HF Transformers 和 ViT Ulysses 集成。

## 不适合或收益较弱的场景

- Decode 每步 `q_len≈1`：没有 Query sequence 可分，Ulysses 的 sequence parallel 基本失去意义；
- 短 Prompt/短训练序列：两次 All-to-All 可能比本地 Attention 更贵；
- `SP > num_q_heads` 或 Q heads 不能整除 SP；
- All-to-All 网络弱、跨节点拓扑不理想；
- KV heads 极少且复制成本不可接受。

## GQA/MQA 是否完全不能用

不是。当前 DeepSpeed HF Ulysses 实现：

```text
要求 num_q_heads % SP == 0
KV heads 则要求 Hkv % SP == 0 或 SP % Hkv == 0
```

当 `SP > Hkv` 时会复制 KV heads。官方源码示例：

```text
Hkv=4, SP=8 → KV heads 复制2倍
```

因此：

- MHA：通常最自然，Q/KV heads 都容易切；
- GQA：可以使用，但 SP 超过 KV-head 数后会复制 KV；
- MQA：也可通过复制单 KV head 适配，但 KV 侧不再获得理想分片收益；
- MLA：latent KV 不是标准 per-head K/V，通常需要特殊适配、物化或复制，不是标准 Ulysses 最自然的对象。

## 为什么 Ring Attention 看起来更常用于长上下文

| 维度 | Ulysses | Ring Attention |
| --- | --- | --- |
| Q | All-to-All 后每 rank 有完整 sequence 的部分 heads | 每 rank 固定本地 Q chunk |
| K/V | All-to-All 后每 rank 有完整 sequence 的部分 heads | K/V blocks 多轮轮转 |
| Softmax | 本地完整 Attention | distributed online softmax |
| 并行上限 | 受 Q heads 可切分性限制 | 不强依赖 head 数 |
| 通信 | 前后明确的 All-to-All 阶段 | 多轮 P2P，可流水重叠 |
| 更自然场景 | 长序列训练、强 All-to-All | 极长 context、少 KV-head、低峰值 KV |

对于 MLA/MQA，Ring 不要求把单个 latent/KV head分给多个 head ranks，因此更容易把 context parallel degree 拉高。代价是通信轮数、online-softmax 状态和 causal 负载均衡更复杂。

## DeepSeek 模型为什么不代表 Ulysses 没意义

1. Ulysses 不是 DeepSeek 的“自研部署方案”，而是 DeepSpeed 的通用训练能力。
2. 它的目标用户包括 GPT/MHA、Llama 类 GQA、长文本 SFT、ViT/视频模型等，不以 MLA 为唯一目标。
3. 即使是 GQA/MQA，当前实现也能通过 KV replication 支持，只是效率边界不同。
4. DeepSeek MLA 若需要极长序列训练，可以采用特殊 Ulysses 适配、Ring/Hybrid CP 或其它 sequence parallel；“标准 Ulysses 不自然”不等于“sequence parallel 无用”。
5. vLLM serving PCP 与 DeepSpeed Ulysses 是两个不同系统：当前 vLLM PCP 没有实现 Ulysses，采用 MLA AllGather 路径，并把 Ring 列为 active-development 方向。

## 官方来源

- [DeepSpeed Ulysses 长序列训练教程](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/docs/_tutorials/ds-sequence.md)
- [DeepSpeed Ulysses HF/ALST 教程](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/docs/_tutorials/ulysses-alst-sequence-parallelism.md)
- [DeepSpeed 当前 Ulysses HF 实现](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L117-L150)
- [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)

## 待核实

DeepSeek 具体训练版本是否使用过 Ulysses 或某种 hybrid context parallel，必须依赖其官方训练报告或源码；本报告不从模型结构反推内部训练拓扑。DeepSpeed 教程中的百万级 token 能力与性能数字也应绑定硬件、模型、checkpoint/offload 配置和具体版本。
