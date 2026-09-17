# SGLang 的 varlen metadata 如何映射到物理 KV slot？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 其他问法

- 调用 varlen attention kernel 时，metadata 怎么组织？
- VRAM data、显存物理切分和 physical slot 是什么关系？

## 30 秒回答

varlen batch 通常先把各请求本轮 token 打平成一维 packed tokens，再用每请求长度和前缀和描述边界。SGLang 的 `ForwardBatch` 中可看到 `seq_lens`、`extend_seq_lens`、`extend_prefix_lens`、`extend_start_loc`、`req_pool_indices` 和 `out_cache_loc`。`req_pool_indices` 选中 `ReqToTokenPool` 的请求行，逻辑位置 `p` 通过 `req_to_token[req_idx,p]` 得到 physical KV slot；新 token 则由 `out_cache_loc` 指定写入 slot。attention backend 再把这些信息转换成自己需要的 `indptr/block table/kv_indices`。

## 深入解释

固定三个层次最容易理解：

```text
packed token 下标
  --extend_start_loc/lengths--> (request, logical_position)
  --req_to_token-------------> physical_slot
  --KV tensor stride/layout--> 实际 K/V 地址
```

例如两个请求本轮分别新增 `[3, 2]` 个 token，则 packed 输入长度为 `5`，`extend_start_loc=[0,3]`。假设它们的 `req_pool_indices=[7,11]`，逻辑位置分别是 `[5,6,7]` 与 `[20,21]`，最终 slot 来自：

```text
req_to_token[7, 5:8]
req_to_token[11, 20:22]
```

这些值可能是 `[64,65,66]` 与 `[208,209]`，kernel 不要求两个请求在物理上相邻。paged 模式下若 page size 为 `P`，还可以分解为：

```text
page_id = slot // P
offset  = slot % P
```

## 常见误区

- varlen metadata 描述 ragged batch 边界；它不是 KV 数据本身。
- logical position、packed token index、KV slot 和 CUDA thread index 是四套不同索引。
- 不同 attention backend 的最终 metadata 名称可能不同，不能把 FlashInfer/Triton/CUTLASS 的结构名当成 SGLang 永久 ABI。

## 关联知识

- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]
- [[../../../../wiki/concepts/KV Cache|KV Cache]]
- [[../../../../wiki/entities/SGLang|SGLang]]

## 参考来源与待核实

- [SGLang `ForwardBatch` metadata](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/model_executor/forward_batch_info.py)
- [SGLang `ReqToTokenPool`](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/memory_pool.py)
- [SGLang paged allocator](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/allocator/paged.py)
- 字段依据 commit `575759d9`；具体 backend 生成的 `qo_indptr/kv_indptr/block table` 需按部署配置继续追源码。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
