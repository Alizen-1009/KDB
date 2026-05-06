---
title: "推理长序列利器：ChunkedPrefill&FlashDecoding原理详解"
source: "https://zhuanlan.zhihu.com/p/1988996116017086993"
author:
  - "[[kaiyuan​新知答主]]"
published:
created: 2026-04-29
description: "在LLM 长序列推理中，常会用到分块预填充(Chunked Prefill)和快速解码(Flash Decoding)来提升性能。虽然两个特性都是将长序列进行拆分计算，但两者的应用场景和计算方式差异较大。本文主要带读者了解ChunkedPrefil…"
tags:
  - "clippings"
---
[收录于 · LLM推理基础与框架](https://www.zhihu.com/column/c_1916901019268391457)

98 人赞同了该文章

目录

收起

1 序列分块运算

2 Chunked Prefill

2.1 应用场景

2.2 代码辅助理解

3 Flash Decoding

3.1 原理介绍

3.2 块内运算的公式

3.3 块间的序列并行

3.3 代码实践

附1: FA分块运算等价证明

在LLM **长序列** 推理中，常会用到分块预填充(**Chunked Prefill**)和快速解码 **(Flash Decoding**)来提升性能。虽然两个特性都是将长序列进行拆分计算，但两者的应用场景和计算方式差异较大。本文主要带读者了解ChunkedPrefill和FlashDecoding的基本原理，通过示例代码理解两种 **分块运算** 的计算过程 **。**

**简单对比：**

|  | ChunkedPrefill | FlashDecoding |
| --- | --- | --- |
| 原理 | 序列并行 + KV cache | Online Softmax |
| 阶段 | prefill，输入长序列 | decoding，输出长序列 |
| 作用层 | 所有层 | [Attention层](https://zhida.zhihu.com/search?content_id=268330210&content_type=Article&match_order=1&q=Attention%E5%B1%82&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzYsInEiOiJBdHRlbnRpb27lsYIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjgzMzAyMTAsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Rlpd7Zn5NdUp162Gbz6x6N2RqZGHeUad3FGgYiUZAFI&zhida_source=entity) |
| 实现方式 | 修改调度器Scheduler | 使用： [FA算子](https://zhida.zhihu.com/search?content_id=268330210&content_type=Article&match_order=1&q=FA%E7%AE%97%E5%AD%90&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzYsInEiOiJGQeeul-WtkCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI2ODMzMDIxMCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.zz1NiGdKQ4Vl_7Va2CcjbE3YWmdNNN8vpOQ9hBidxqM&zhida_source=entity) +分块运算 |
| 直接影响 | 降低单次的计算量&显存 | 提升Attention运算并发度 |
| 效果 | 预填充与解码混合下发，减少空泡 | 并行计算，提升GPU利用率 |

本文相关代码地址 [^1] ： [github.com/CalvinXKY/In](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/chunked_prefill_and_flash_decoding.ipynb)

## 1 序列分块运算

首先，明确实现分布式推理需要考量的核心问题：序列并行的计算结果是否具备等价性？

若将长序列切分为多个子序列，由不同 GPU 分别计算得到子输出后直接拼接形成最终结果，该方式的计算结果，与对完整长序列直接计算所得的结果是否一致？

![](https://pic1.zhimg.com/v2-a2ecd754573eae993e4d56e02c48fa90_1440w.jpg)

这个问题在 [《LLM推理并行优化的必备知识 》](https://zhuanlan.zhihu.com/p/1937449564509545940) 中有过分析，现直接引用其结论：

- 对于线性层、归一化层、embedding等层，结果 **相等** ；
- 对于attention层，因为有softmax计算，所以直接拼接结果 **不相等** 。

对于第二点的进一步分析结论：

- Q支持直接切分序列，而KV不能直接切分序列。
- KV序列并行运算，需要修正softmax的计算结果。

Chunked Prefill和Flash Decoding都结合了序列并行(Sequence Parallel)的原理，两者实现存在较大差异：

- Chunked Prefill：所有层都进行了序列并行，由于有 **KV cache** 使得KV值保持完整，所以Attention计算不受影响。
![](https://pic4.zhimg.com/v2-b7100f9847fc7f9f9078eb136b1f7071_1440w.jpg)

- Flash Decoding：针对Attention KV cache的分块运算。块内采用FlashAttention计算，块间采用序列并行。

需要注意的是，这两个特性一般是针对 **GPT类模型** ，即Attention为因果运算模式。

## 2 Chunked Prefill

在长序列推理请求中，Chunked Prefill 机制能够有效削减单次 Prefill 过程的计算开销与峰值显存需求，从而优化系统资源分配，提升整体资源利用率。

### 2.1 应用场景

**一、适配超长prompt**

![](https://pic4.zhimg.com/v2-bd936ece6aa9681660fa8a47b492aca5_1440w.jpg)

分块下发：1->2->3

该方式能解决因输入序列过长、GPU资源不足导致单请求无法执行的问题。

需注意的是，序列分段的下发顺序不能随意调整， **必须依次下发** 。

**二、与decode混合执行，降空泡**

![](https://pic2.zhimg.com/v2-a623b702c5968386e7f6e3771ec0b33b_1440w.jpg)

Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

将prefill拆分成多个子块，并与其它请求的decode一起执行 [^2] ，这种方式能够有效地降低资源空泡（bubble），提升整体的资源利用率。过程原理示意参考NVIDIA的示例 [^3] （由

[@Faded](https://www.zhihu.com/people/755ae95f79da4f6435389c55be7bf5b2)

推荐）：

![](https://picx.zhimg.com/v2-a365c2ce62b463e5490b80378a6b1dcd_1440w.jpg)

NVIDIA TensorRT-LLM Chunked Prefill

**特性的实现** ：在推理框架中，主要改动点是调度器(Scheduler)的逻辑，保证多次分块的prefill请求能够衔接起来。

调度器的输出格式：{ **请求序号** ： **tokens数量** }

通过控制每个请求本轮需要计算的tokens数量，实现任意chunk大小的组合下发。

![](https://pic1.zhimg.com/v2-fe2a02768f801255b3acd115fbd22b72_1440w.jpg)

vLLM框架中的应用

### 2.2 代码辅助理解

在Attention层计算场景下，构建Chunked Prefill流水输出机制，同时引入标准Prefill作为对照组，开展计算结果的对比与校准工作。

标准的MHA模块Prefill运算：

```python
class Prefill(nn.Module):
    """
    流式 + 因果的Chunked Prefill实现
    专为自回归LLM（如GPT、LLaMA）的推理优化
    """
    
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将张量分割成多头"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将多头合并"""
        batch_size, n_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    def prefill_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准注意力（不分块）- 用于验证正确性
        因果注意力：每个位置只能看到之前的位置
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分割多头
        q = self._split_heads(q)  # [batch, n_heads, seq_len, head_dim]
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用因果掩码（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        output = self._merge_heads(attn_output)
        output = self.out_proj(output)
        
        return output
```

构造支持chunked prefill的MHA。

为了演示方便，采用for循环 **模拟分chunk下发执行，** 最后进行输出合并。

```python
# 将下面函数放入类中：
    def prefill_chunked(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Args:
            x: 输入序列 [batch, seq_len, d_model]
            
        Returns:
            output: 注意力输出 [batch, seq_len, d_model]
            kv_cache: KV缓存 (K列表, V列表)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算总chunk数
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # 初始化KV缓存（存储每个chunk的K和V）
        k_cache = []  # 每个元素: [batch, n_heads, chunk_size, head_dim]
        v_cache = []  # 每个元素: [batch, n_heads, chunk_size, head_dim]
        
        # 存储每个chunk的输出
        outputs = []
        
        print(f"分块预填充: 序列长度={seq_len}, 分块大小={self.chunk_size}, 分块数={n_chunks}")
        
        
        for chunk_idx in range(n_chunks):
            # 当前chunk的起始和结束位置
            start = chunk_idx * self.chunk_size
            end = min((chunk_idx + 1) * self.chunk_size, seq_len)
            chunk_len = end - start
            
            # 获取当前chunk
            chunk = x[:, start:end, :]
            
            # 计算当前chunk的QKV
            q = self.q_proj(chunk)
            k = self.k_proj(chunk)
            v = self.v_proj(chunk)
            
            # 分割多头
            q = self._split_heads(q)  # [batch, n_heads, chunk_len, head_dim]
            k = self._split_heads(k)
            v = self._split_heads(v)
            
            # 将当前chunk的K和V添加到缓存
            k_cache.append(k)
            v_cache.append(v)
            
            # 当前累计的KV总长度
            total_kv_len = sum(k.shape[2] for k in k_cache)
            
            # 拼接当前所有可用的K和V（因果：只能看到当前和之前的chunk）
            k_all = torch.cat(k_cache, dim=2)  # [batch, n_heads, total_kv_len, head_dim]
            v_all = torch.cat(v_cache, dim=2)
            
            # 计算注意力分数
            scores = torch.matmul(q, k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # 创建因果掩码
            # 注意：我们需要确保当前chunk内的Q也不能看到同一chunk内未来的K
            # 所以需要构建一个 [chunk_len, total_kv_len] 的掩码
            
            # 方法1：构建完整的掩码矩阵
            q_positions = torch.arange(chunk_len, device=x.device).unsqueeze(1) + start
            kv_positions = []
            for i, k_chunk in enumerate(k_cache):
                kv_start = i * self.chunk_size
                kv_len = k_chunk.shape[2]
                kv_positions.extend(range(kv_start, kv_start + kv_len))
            kv_positions = torch.tensor(kv_positions, device=x.device).unsqueeze(0)
            
            # Q位置只能看到小于等于它的KV位置
            mask = q_positions >= kv_positions  # [chunk_len, total_kv_len]
            mask = mask.view(1, 1, chunk_len, total_kv_len)
            
            # 应用掩码
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # 注意力输出
            attn_output = torch.matmul(attn_weights, v_all)
            
            # 合并多头
            output_chunk = self._merge_heads(attn_output)
            output_chunk = self.out_proj(output_chunk)
            
            outputs.append(output_chunk)
            
            print(f"  处理chunk {chunk_idx+1}/{n_chunks}: "
                  f"位置 {start}:{end}, "
                  f"KV缓存长度={total_kv_len}")
        
        # 拼接所有chunk的输出
        output = torch.cat(outputs, dim=1)
        
        return output, (k_cache, v_cache)
```

测试用例如下：

```python
def test():
    # 创建模型
    model = CausalChunkedPrefill(
        d_model=d_model,
        n_heads=n_heads,
        chunk_size=chunk_size
    )
    x = torch.randn(batch_size, seq_len, d_model)
   
    # 1. 标准注意力（不分块）
    print("\n1. 计算标准注意力（不分块）...")
    with torch.no_grad():
        output_standard = model.prefill_standard(x)
    print(f"   标准注意力输出形状: {output_standard.shape}")
    
    # 2. 分块预填充
    print("\n2. 计算分块预填充...")
    with torch.no_grad():
        output_chunked, kv_cache = model.prefill_chunked(x)
    print(f"   分块预填充输出形状: {output_chunked.shape}")
    
    # 3. 比较结果
    print("\n3. 比较两种方法的输出...")
    diff = torch.abs(output_standard - output_chunked)
```

打印可看到，两者结果是一致的。

详细的代码参考： [chunked\_prefill\_and\_flash\_decoding.ipynb](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/chunked_prefill_and_flash_decoding.ipynb) 第1节。

## 3 Flash Decoding

Flash Decoding作为一种推理解码阶段的加速方案，其基础思想已被许多现代推理加速库（如 [FlashInfer](https://zhida.zhihu.com/search?content_id=268330210&content_type=Article&match_order=1&q=FlashInfer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzYsInEiOiJGbGFzaEluZmVyIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjY4MzMwMjEwLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.mzmIhu9O-DkJGJwTqYcNdS2vo_dPrrlW_6cCEyCYHKg&zhida_source=entity) [^4] 、 [Triton](https://zhida.zhihu.com/search?content_id=268330210&content_type=Article&match_order=1&q=Triton&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzYsInEiOiJUcml0b24iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjgzMzAyMTAsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.RIlaPSLk1ZjjqB6vJzqz_450DJxebtgI0OFDIQOOhbc&zhida_source=entity) ）所吸收和扩展。在实际框架中，通常调用的是在其基础上进行了进一步优化的算子（例如，FlashInfer 通过块稀疏KV缓存、可定制注意力模板和负载平衡调度实现了更优的性能），而非最原始的版本。后续出现的FA技术也可认为是这一方向的演进。因此，学习Flash-Decoding 的核心价值在于理解其 **分块计算与归约的基本原理** ，这是掌握当前解码加速技术的基础。

### 3.1 原理介绍

在推理的decode阶段，当batch size比较小时，attention运算阶段的GPU利用率低 [^5] 。具体而言，就是Q值batch size小于GPU的处理单元SM数量，如果串行计算会导致SM出现空闲。长序列串行带来另一个问题，会使得推理的TPOT过大。

![动图封面](https://pic3.zhimg.com/v2-ed9b861913c09f2d8d6893bb5b63580a_b.jpg)

串行执行

Flash Decoding解决方式是让KV进行分块。 每个块内用FA计算，块间结果最后进行一次规约运算。如下图所示，拆分成了5个块的decoding。

![动图封面](https://pic4.zhimg.com/v2-a4fea9764b0988e261c9bcda7297b837_b.jpg)

并发执行

从数据加载角度看，每个SRAM上面加载一份Q数据，然后依次加载Split 块分配的KV值。

![](https://pic4.zhimg.com/v2-0e9d9a2fba22c8ae66fdd4113037f395_1440w.jpg)

数据分配示意图

若KV Cache分布式部署于多个GPU节点，Flash Decoding的实现逻辑依然成立，仅需新增集合通信环节，完成对Q数据的分发，以及O、S两类数据的跨卡交互（其中S为log-sum-exp变量）。

![](https://pic2.zhimg.com/v2-731f95568a7ed08c2a4496457a08700d_1440w.jpg)

分布式 KV cache的Flash Decoding

### 3.2 块内运算的公式

在推理的Decoding阶段，对每个请求而言输入token长度皆为1。这个特点使得只需要用到FlashAttention公式Forward过程的 **内层循环** [^6] 。

![](https://pic2.zhimg.com/v2-30d6c3f8d18eb86e3b924a2463c1f945_1440w.jpg)

FlashAttention2 论文公式

在Online Softmax [^7] 中有个更简洁的公式版本。

![](https://pic3.zhimg.com/v2-62235dd6f041a67761d693306adc3bd8_1440w.jpg)

FA的计算公式

若用for循环来模拟多次数据加载，实现代码如下：

```python
def flash_decoding_without_cp(self, q, k, v, block_size=32):
    """
    分块的FA

    """
    batch_size, num_heads, seq_len_q, _ = q.shape
    seq_len_kv = k.shape[2]
    num_blocks = (seq_len_kv + block_size - 1) // block_size

    # 初始化累积变量
    # 累积的加权和
    numerator = torch.zeros(batch_size, num_heads, seq_len_q, self.head_dim,
                            device=q.device, dtype=q.dtype)
    # 累积的归一化因子
    d_prime = torch.zeros(batch_size, num_heads, seq_len_q, 1,
                              device=q.device, dtype=q.dtype)

    # 用于数值稳定性的全局最大值
    global_max = torch.full((batch_size, num_heads, seq_len_q, 1),
                            -float('inf'),
                            device=q.device, dtype=q.dtype)

    # 分块处理
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, seq_len_kv)

        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]

        # 计算当前块的注意力分数
        scores_block = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 当前块的最大值
        block_max = scores_block.max(dim=-1, keepdim=True).values

        # 更新全局最大值
        # 我们需要比较每个位置（每个query）在所有块中的最大值
        new_global_max = torch.maximum(global_max, block_max)

        # 调整之前累积的权重（基于新的全局最大值）
        # 当全局最大值更新时，需要重新调整之前累积的权重
        if block_idx > 0:
            # 将之前积累的权重调整到新的尺度
            adjustment_factor = torch.exp(global_max - new_global_max)
            numerator = numerator * adjustment_factor
            d_prime = d_prime * adjustment_factor

        # 更新全局最大值
        global_max = new_global_max

        # 计算当前块的指数权重（减去全局最大值以保持数值稳定）
        exp_scores = torch.exp(scores_block - global_max)
        block_sum_exp = exp_scores.sum(dim=-1, keepdim=True)

        # 累积加权和
        numerator = numerator + torch.matmul(exp_scores, v_block)
        d_prime = d_prime + block_sum_exp

    # 最终归一化
    final_output = numerator / d_prime
    return final_output
```

### 3.3 块间的序列并行

序列分块运算每个块对应都获得了一个O输出，通过上述代码公式，可知这些局部的O结果中

- block\_sum\_exp，即softmax运算的分母exp求和为局部值，非全序列下所有值求的和；
- global\_max，为局部最大score，非全序列下的最大值。

若将所有分块O值直接求和，得不到最终结果。正确的做法是先对分块O值进行 **修正** 再求和。在blockwise transformer [^8] 、context parallel [^9] 等中有相关方法的介绍。这里分析两种：

**方式一：** 保存每个块的max值、block\_sum\_exp值。步骤：

- 先用FA算完所有分块的局部结果；
- 找到全局max值；
- 用全局max值修正block\_sum\_exp值；
- 合并归一化因子；
- 合并输出，完成最终的归一化。
![](https://pic3.zhimg.com/v2-514cf7702ff319577b283a45d55fff50_1440w.jpg)

代码实现如下：

```python
def flash_decoding_attention_simple(self, q, k, v, block_size=32):

    batch_size, num_heads, seq_len_q, _ = q.shape
    seq_len_kv = k.shape[2]
    num_blocks = (seq_len_kv + block_size - 1) // block_size

    # 存储每个块的中间结果
    block_outputs = []
    block_max_vals = []
    block_sum_exps = []

    # 第一步：计算每个块的局部结果
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, seq_len_kv)

        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]

        # 计算当前块注意力分数
        scores_block = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
        block_max = scores_block.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores_block - block_max)
        block_sum_exp = exp_scores.sum(dim=-1, keepdim=True)

        # 存储中间结果
        block_outputs.append(torch.matmul(exp_scores, v_block))
        block_max_vals.append(block_max)
        block_sum_exps.append(block_sum_exp)

    # 第二步：合并所有块的结果
    # 找到全局最大值
    all_max_vals = torch.stack(block_max_vals, dim=0)  # [num_blocks, ...]
    global_max = all_max_vals.max(dim=0).values  # 在每个query位置取最大值

    # 合并归一化因子
    total_sum_exp = torch.zeros_like(block_sum_exps[0])
    for i in range(num_blocks):
        total_sum_exp += block_sum_exps[i] * torch.exp(block_max_vals[i] - global_max)

    # 合并输出
    final_output = torch.zeros_like(block_outputs[0])
    for i in range(num_blocks):
        # 将每个块的贡献调整到全局尺度
        weight = torch.exp(block_max_vals[i] - global_max)
        final_output += block_outputs[i] * weight

    # 最终归一化
    final_output = final_output / total_sum_exp

    # 计算完整注意力权重用于验证
    full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    full_attention_weights = F.softmax(full_scores, dim=-1)

    return final_output, full_attention_weights
```

**方式二：** 保存每个块的log-sum-exp值

这种方式相较于方式一，需要存储的内容更少，计算复杂度却不会增加。两个分块的合并运算，引用FlashInfer中的定义 [^10] ：

![](https://pic4.zhimg.com/v2-0b39872c6ecdb48cbeb31c81d882790f_1440w.jpg)

其中LSE(Log-Sum-Exp)的计算： ，与FA2的forward的计算相同，只是在FA2中是先减去max值，计算LSE时再加回来。

与方式一的等价证明见 **附件1** 。多个分块的修正合并运算的代码如下：

```python
def merge_streams_two_step(self, streams_data):
    """
    两步合并算法：
    1. 迭代计算全局 S_global
    2. 用 S_global 修正每个流的输出贡献
    """
    if not streams_data:
        return None
    
    # 提取所有流的S_i
    S_list = [S_i for _, S_i in streams_data]
    
    # 步骤1: 迭代计算全局 S_global (S_lst)
    S_global = S_list[0].clone()
    
    for i in range(1, len(S_list)):
        S_i = S_list[i]
        S_max = torch.maximum(S_global, S_i)
        S_min = torch.minimum(S_global, S_i)
        # 使用log(1+exp(x))的稳定计算
        log_term = torch.log1p(torch.exp(S_min - S_max))
        S_global = S_max + log_term
    
    # 步骤2: 修正每个流的输出贡献
    O_global = torch.zeros_like(streams_data[0][0])
    
    for O_i, S_i in streams_data:
        # 计算该流对全局的贡献权重
        weight = torch.exp(S_i - S_global)
        # 累加加权贡献
        O_global += O_i * weight
    
    return O_global
```

### 3.3 代码实践

选用标准的attention运算中scaled dot-product部分作为参考基线。实现如下：

```python
import torch
import torch.nn.functional as F
import math
import time

# 创建一个类，定义如下
class FlashDecodingDemo:
    def __init__(self, d_model: int = 64, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def traditional_attention(self, q, k, v):
        """传统连续注意力计算"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
```

这里的F.softmax为torch内部safe softmax，实际上减去了最大值。等价于如下步骤：

1. `m = max(x)`
2. `x_shifted = x - m`
3. `exp_x = exp(x_shifted)`
4. `softmax = exp_x / sum(exp_x)`

FA在分块运算上面，为了方便演示采用for循环的方式模拟多设备运算。

```python
def flash_decoding_with_lse(self, q, k, v, 
                            tile_size_kv: int = 256,
                            num_streams: int = 4):
        """
        Flash-Decoding 仅存储O和S
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
        
        print(f"使用两步合并算法Flash-Decoding: {num_streams}个流")
        
        # 初始化流数组
        streams_data = []
        
        for stream_id in range(num_streams):
            # 每个流存储(O_i, S_i)
            O_stream = torch.zeros_like(q)
            S_stream = torch.full((batch_size, num_heads, seq_len_q, 1), 
                                -float('inf'), device=q.device, dtype=q.dtype)
            streams_data.append((O_stream, S_stream))
        
        # 处理每个tile
        print(f"处理{num_tiles}个tile...")
        
        for tile_idx in range(num_tiles):
            stream_id = tile_idx % num_streams
            
            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)
            
            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]
            
            # 计算当前tile的输出
            O_i, S_i = self.compute_stream_output(q, k_tile, v_tile)
            
            # 获取当前流的累加器
            O_acc, S_acc = streams_data[stream_id]
            
            # 合并当前tile结果到流累加器
            if torch.all(S_acc == -float('inf')):
                streams_data[stream_id] = (O_i, S_i)
            else:
                # 使用两步法合并当前tile到流累加器
                # 先计算合并后的S
                S_max = torch.maximum(S_acc, S_i)
                S_min = torch.minimum(S_acc, S_i)
                log_term = torch.log1p(torch.exp(S_min - S_max))
                S_merged = S_max + log_term
                
                # 修正两个部分的贡献
                weight_acc = torch.exp(S_acc - S_merged)
                weight_i = torch.exp(S_i - S_merged)
                O_merged = O_acc * weight_acc + O_i * weight_i
                
                streams_data[stream_id] = (O_merged, S_merged)
        
        print(f"所有tile处理完成，开始归约所有流...")
        # 归约所有流的结果
        O_final = self.merge_streams_two_step(streams_data)
        
        return O_final
```

构建测试，能够验证Flash Decoding的正确性：

![](https://pica.zhimg.com/v2-6a82a762a4f2427b2b9b7bd9bee949b4_1440w.jpg)

详细的代码参考\[1\]： [chunked\_prefill\_and\_flash\_decoding.ipynb](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/chunked_prefill_and_flash_decoding.ipynb) 第2节。

### 附1: FA分块运算等价证明

**一、计算方式的区别**

1、Attention的一般计算方式：

2、 Flash-Decoding分块计算，将N个分数分成K个块，每个块i有：

**二、合并方式的区别**

1、传统合并算法步骤：

- 找到全局最大值
- 调整每个块的贡献：
- 合并：

2、分块合并算法步骤：

**三、 等价证明**

**步骤1** ：证明

**步骤2** ：证明

**步骤3** ：证明

**步骤4** ：证明

**步骤5** ：证明 等于传统合并结果

---

**vLLM的框架其它内容参看：**

**欢迎点赞、关注、留言讨论。**

[@kaiyuan](https://www.zhihu.com/people/da4e6b50eb50d6f120b604f6cf15b33e)

## 参考

编辑于 2026-03-26 19:39・中国香港[大模型](https://www.zhihu.com/topic/25402720)[人工智能](https://www.zhihu.com/topic/19551275)[LLM](https://www.zhihu.com/topic/20660508)

[^1]: [https://github.com/CalvinXKY/InfraTech/blob/main/llm\_infer/chunked\_prefill\_and\_flash\_decoding.ipynb](https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/chunked_prefill_and_flash_decoding.ipynb)

[^2]: [https://arxiv.org/pdf/2308.16369](https://arxiv.org/pdf/2308.16369)

[^3]: [https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)

[^4]: [https://www.arxiv.org/pdf/2501.01005](https://www.arxiv.org/pdf/2501.01005)

[^5]: [https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

[^6]: [https://arxiv.org/pdf/2307.08691](https://arxiv.org/pdf/2307.08691)

[^7]: [https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

[^8]: [https://arxiv.org/abs/2305.19370](https://arxiv.org/abs/2305.19370)

[^9]: [https://zhuanlan.zhihu.com/p/698447429](https://zhuanlan.zhihu.com/p/698447429)

[^10]: [https://www.arxiv.org/pdf/2501.01005](https://www.arxiv.org/pdf/2501.01005)