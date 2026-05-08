---
title: "PageAttention代码走读"
source: "https://zhuanlan.zhihu.com/p/668736097"
author:
  - "[[zzk againAbove & Beyond]]"
published:
created: 2026-05-08
description: "后续有更多博客补档前言vLLM是一款性能优秀，自带动态插入方案的LLM推理方案，之前本来想写一篇vLLM的文章，但是知乎上已经有很多优秀的博客了就不再赘述，本篇文章就讲讲其中的一个PageAttention CUDA算子 之前就…"
tags:
  - "clippings"
---
[收录于 · 深度学习框架开发专栏](https://www.zhihu.com/column/c_1304184712240340992)

DefTruth 等 211 人赞同了该文章

> 后续有更多博客补档

### 前言

[vLLM](https://zhida.zhihu.com/search?content_id=236738446&content_type=Article&match_order=1&q=vLLM&zhida_source=entity) 是一款性能优秀，自带动态插入方案的LLM推理方案，之前本来想写一篇vLLM的文章，但是知乎上已经有很多优秀的博客了就不再赘述，本篇文章就讲讲其中的一个PageAttention [CUDA算子](https://zhida.zhihu.com/search?content_id=236738446&content_type=Article&match_order=1&q=CUDA%E7%AE%97%E5%AD%90&zhida_source=entity)

之前就想试图读懂 [FasterTransformer](https://zhida.zhihu.com/search?content_id=236738446&content_type=Article&match_order=1&q=FasterTransformer&zhida_source=entity) 的decoder masked multihead attention，但由于逻辑过分复杂，尝试了很多次也无法特别清楚理解。

PageAttention修改自FT的kernel，并精简了相关逻辑（没有把融合做的那么变态），本篇博客就简单解析下

### PageAttention原理

在自回归解码过程中，query会不断和cachekv进行交互做注意力机制。而FasterTransformer里面为了避免重复的数据搬运（即把新kv和历史kv concat），预先会分配出 max\_seq\_len 长度的cachekv，后续只需要往里面写入即可，节省了昂贵的数据搬运操作

而在实际场景中，推理往往以变长为主，不同query生成的长度不一，粗暴的将cachekv按 max\_seq\_len 长度分配很容易造成显存的浪费。

那么PageAttention正是为了改进这一点而生（当然其中一个原因也包括他的动态插入逻辑），一个query对应的CacheKV不一定需要连续的显存， **他将连续个token存储的CacheKV划分为一个block** ，并且有一个block\_tables维护各个query对应CacheKV是哪几个block，进而索引

vLLM官方博客 **[vllm.ai](https://link.zhihu.com/?target=https%3A//vllm.ai/)** 里有个动图很生动形象的解释了PageAttention

![动图](https://pic3.zhimg.com/v2-e8a2317d1bc7ba5670ca05f68196453e_b.webp)

简单介绍完原理后我们直接看代码： **[attention\_kernels.cu](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu)**

### Dispatch逻辑

- CALL\_KERNEL\_LAUNCHER\_BLOCK\_SIZE 根据存储的kv blocksize进行派发，分别是 8， 16， 32
- LAUNCH\_ATTENTION\_KERNEL 根据注意力头大小HEADSIZE静态派发

### Kernel Launch参数

- dim3 grid(num\_heads, num\_seqs);
- dim3 block(NUM\_THREADS);

其中NUM\_THREADS固定为128

### Kernel 输入参数

- out \[num\_seqs, num\_heads, head\_size\]
- q \[num\_seqs, num\_heads, head\_size\]
- k\_cache \[num\_blocks, num\_kv\_heads, head\_size/x, block\_size, x\] 这里的x表示一个向量化的大小，如float16 -> 16 / sizeof(float16) = 8
- v\_cache \[num\_blocks, num\_kv\_heads, head\_size, block\_size\]
- head\_mapping \[num\_heads\] 用于MQA, GQA，确定用的KV\_head
- block\_tables \[num\_seqs, max\_num\_blocks\_per\_seq\] block\_tables映射表，表示每个sequence映射到哪几个block上
- context\_lens \[num\_seqs\] 用于变长

### Kernel的一些常量定义

- THREAD\_GROUP\_SIZE = MAX(WARP\_SIZE / BLOCK\_SIZE, 1) 通过WARPSIZE / BLOCKSIZE 得到一个thread\_group大小。注意这里的BLOCKSIZE不是cuda blocksize，而是一个kv block的大小
- NUM\_TOKENS\_PER\_THREAD\_GROUP = (BLOCK\_SIZE + WARP\_SIZE - 1) / WARP\_SIZE 表示每个thread\_group处理多少个token
- NUM\_WARPS 表示一个threadblock有多少个warp
- VEC\_SIZE 表示向量化大小，保证每个thread\_group一次性获取16bytes，MAX(16 / (THREAD\_GROUP\_SIZE \* sizeof(scalar\_t)), 1);

> 这里比较疑惑的是为什么让一个thread\_group读取16bytes，而不是一个thread读取16bytes

- NUM\_ELEMS\_PER\_THREAD = HEAD\_SIZE / THREAD\_GROUP\_SIZE 表示每个thread要负责多少个数据计算
- NUM\_VECS\_PER\_THREAD = NUM\_ELEMS\_PER\_THREAD / VEC\_SIZE; 表示每个thread负责的数据经过向量化后，一共有多少个vec
- V\_VEC\_SIZE = MIN(16 / sizeof(scalar\_t), BLOCK\_SIZE) 每个thread一次性读取16bytes
- NUM\_V\_VECS\_PER\_ROW = BLOCK\_SIZE / V\_VEC\_SIZE。对于v\_cache\[head\_size, block\_size\]，表示一行需要几个V\_VEC
- NUM\_ROWS\_PER\_ITER = WARP\_SIZE / NUM\_V\_VECS\_PER\_ROW 表示一个warp可以处理多少行
- NUM\_ROWS\_PER\_THREAD 表示每个thread需要负责多少行

### 代码走读

### part1 加载Query

根据前面得到每个线程要处理的vector个数，以及thread\_group\_idx进行偏移，获取数据：

```
const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  Q_vec q_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
```

### part2 申请shared\_memory

```
// Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];
```

shared\_memory分为两部分：

- 一部分用于存储QK结果来做softmax
- 另一部分是给blockReduce的smem使用

### part3 提前偏移block tables等参数

```
// x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  x 表示每个thread_group一次性取这么多个key
  constexpr int x = 16 / sizeof(scalar_t);
  // 提前设置为-max，用于后面维护max值
  float qk_max = -FLT_MAX;
  
  // 拿到block table指针
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  // 拿到实际request的sequence长度
  const int context_len = context_lens[seq_idx];
  // 计算该request需要多少个kv block存储cachekv
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

### part4 循环计算QK

1. 每个warp负责计算一个block key，而每个block key shape为 \[block\_size, num\_head, head\_size\]
2. 每个thread\_group取一个key，即num\_head个元素，计算QK dot
```
//  每个warp负责 blocksize * headsize个元素
for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    // TODO(Zhengzekang)
    const int physical_block_number = block_table[block_idx];
    // ...
    K_vec k_vecs[NUM_VECS_PER_THREAD];
    
    // 遍历每个thread_group处理多少个token
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      
      // ....
      
      //  遍历每个thread需要处理多少个VEC
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      //  vectorized取到key
      k_vecs[j] = xxxx;
  
    // 计算QKdot，里面包含了一个thread_groupsize的WarpReduceSum，
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);

    // 只有thread_group的第一个thread负责将QK结果存储到shared memory
    // 并且维护一个qk_max，用于后续softmax
    if (thread_group_offset == 0) {
      // Store the partial reductions to shared memory.
      // NOTE(woosuk): It is required to zero out the masked logits.
      const bool mask = token_idx >= context_len;
      logits[token_idx] = mask ? 0.f : qk;
      // Update the max value.
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
 }
}
```

此时各个thread\_group已经完成了自己的qk\_dot操作，并且都维护了qk\_max。

那么下面就需要和其他thread\_group做warp shuffle操作，得到一个warp内的qk max值。

而又由于每个thread\_group里的thread内维护的qk\_max是一样的，所以warp shuffle只需到 thread\_group\_size即可停止

并由lane\_id = 0的线程将warp里的qk\_max存储到smem，最后再做一次warpreduce，得到一个block里的qkmax值，通过shfl\_sync广播操作，让每个线程都拿到max

```
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);
```

接下来就是常规的softmax操作了：

```
// Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();
```

### logits dot V\_Cache

在原始attention里，是(1，seqlen) matmul (seqlen, headsize)，如下图所示：

![](https://picx.zhimg.com/v2-f6b10ac0bd8200b96a446055b86680f3_1440w.jpg)

但是为了读写连续，它将V\_cache转置，shape为：\[num\_blocks, num\_kv\_heads, head\_size, block\_size\]，即：

![](https://pic3.zhimg.com/v2-b22e126c37301947d0c937b01cb044d2_1440w.jpg)

```
// 每个线程一次性读16bytes数据
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  
  // 每一行有多少个V_VEC，假设BLOCK_SIZE=8，那么NUM_V_VECS_PER_ROW=1
  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  // 一个WARP一次处理多少行，按照上面假设，这里是32
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  // 每个thread需要负责多少行，假设headsize=128，那么每个thread要处理4行
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // 提前分配accumulate buffer，用float累加
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    // ...
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }
```
![](https://pic2.zhimg.com/v2-cd72f7c5d11d9c96f87fa512cb24a0a7_1440w.jpg)

由于一行可能有多个V\_VEC，而且是由不同thread负责的计算(如果一行有2个V\_VEC，那么thread0计算V\_VEC0, thread1计算V\_CEC1)，所以需要做一个 [warp\_reduce](https://zhida.zhihu.com/search?content_id=236738446&content_type=Article&match_order=1&q=warp_reduce&zhida_source=entity) ，这里的mask就是 NUM\_V\_VECS\_PER\_ROW

```
// Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }
```

### 最终结果更新

这里比较巧妙，将一个block分成上半部分warp和下半部分warp。

- NUM\_WARPS > warp\_id > mid，上半部分warp将自己累加的结果写到shared memory
- mid > warp\_id > 0，下半部分warp将之前上半部分WARP存储到shared\_memory结果取出，进行累加

这样重复，最终warp\_idx=0的warp即可得到最终结果

![](https://pic4.zhimg.com/v2-f64777c16656c2a1057814245ac0e223_1440w.jpg)

### 延伸

vLLM社区里面有一个非常不错的ncu profile教程，里面通过ncu分析，进而优化，将一个block用到的query读到shared memory缓存（尽管FasterTransformer之前已经是这么做的了）

**[+34% higher throughput?](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/421)**

发布于 2023-11-25 12:21・广东[CUDA](https://www.zhihu.com/topic/19597236)