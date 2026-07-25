# 算子与GPU优化、推理优化补充

## 说明

这份文档回答两组高频题：

- 算子与 GPU 优化：`特定 shape 调优`、`算子融合`、`混合精度`、`Nsight 工具链`
- 推理优化：`KV Cache`、`FlashAttention`

风格仍然遵循三件事：

- 先给结论
- 再讲机制
- 最后补 tradeoff 和工程判断

相关概念可结合 [[../wiki/concepts/CUDA Kernel|CUDA Kernel]]、[[../wiki/concepts/算子融合|算子融合]]、[[../wiki/concepts/Profiling|Profiling]]、[[../wiki/concepts/FlashAttention|FlashAttention]]、[[../wiki/concepts/KV Cache|KV Cache]] 一起看。

## 5. 针对特定 shape 的算子调优，如何制定优化策略以超越官方库的性能？

先给结论：

> 想超越官方库，前提通常不是“你写得比 cuBLAS/cuDNN 更聪明”，而是“你的 workload 比官方库更窄、更稳定、更可假设”。真正能赢的点，一般来自 shape 固定、数据布局固定、算子链固定，以及你愿意为了这一小簇 shape 牺牲通用性。

一个实用策略是按下面 6 步走。

### 1. 先确认你到底在和谁比

- 是和 `PyTorch eager` 比，还是和 `torch.compile` 比
- 是和官方通用 kernel 比，还是和已经高度特化的 `cuBLASLt / cuDNN / FlashAttention` 比
- baseline 要固定硬件、驱动、CUDA 版本、输入 shape、warmup 和计时口径

如果 baseline 没设稳，后面“优化”很容易只是测量误差。

### 2. 先做算账：这个 shape 是 memory-bound 还是 compute-bound

先从 [[../wiki/concepts/Roofline 模型|Roofline 模型]] 想：

- 如果算子 arithmetic intensity 很低，优先想 `访存重排 / 融合 / 复用`
- 如果已经接近 compute-bound，优先想 `Tensor Core 命中 / tile 设计 / pipeline overlap`

不要一上来就改 block size。先判断瓶颈方向，优化才不会偏。

### 3. 抓住“固定 shape”给你的额外信息

固定 shape 的价值在于你可以大胆利用官方库不敢假设的条件：

- 维度长期不变，可以做 shape-specialized kernel
- 输入布局固定，可以预先选择最优内存排布
- 上下游算子固定，可以做更深的 epilogue / prologue fusion
- batch、head 数、hidden size 落在少数桶里，可以离线 autotune 后缓存最优配置
- 请求模式稳定，可以用 persistent kernel、预分配 workspace、甚至 CUDA Graphs 降低 launch 开销

这类优化的本质是：把“运行时决策”挪到“编译时或部署时决策”。

### 4. 官方库很难覆盖，但你可以特化的几个常见方向

#### 方向 1：更激进的融合

比如把：

- `matmul + bias + activation`
- `qk^t + mask + softmax + pv`
- `residual + rmsnorm`

合成更少的 kernel，减少中间张量写回。

#### 方向 2：更贴 workload 的 tile 和线程映射

对于固定 `M/N/K`，可以专门挑：

- CTA tile
- warp tile
- MMA tile
- stage 数
- block size

目标不是“通用最优”，而是“这一个 shape 集合最优”。

#### 方向 3：减少通用库的动态分支和通用开销

官方库需要处理：

- 边界维度
- 多种 layout
- 多种 dtype
- 多种 epilogue
- 多种对齐情况

你的内核如果只服务少量 shape，就能删掉很多动态判断，减少指令和控制流分歧。

#### 方向 4：重写数据流，而不是只微调 kernel

很多真正的大收益不是来自某条指令，而是来自：

- 改变张量布局让访存更合并
- 让中间状态留在寄存器或 shared memory
- 用双缓冲、流水化加载隐藏访存延迟
- 在 decode 或小 batch 场景中改成 persistent kernel，减少 launch / tail effect

`FlashAttention` 就是典型例子：它赢的核心不是数学近似，而是 IO-aware 的数据流重排。

### 5. 用 Nsight 验证你到底赢在哪

如果自定义 kernel 真的更快，通常能在 profiler 里看到至少一类变化：

- DRAM 吞吐更接近上限，说明你更 bandwidth-efficient
- Tensor Core / SM pipe 利用率更高，说明你更 compute-efficient
- kernel 数量减少，GPU timeline 更紧凑
- `warp stall` 下降
- occupancy 没一定更高，但“有效利用率”更高

如果只是 runtime 快了一点，但 profiler 看不出原因，通常说明结论还不稳。

### 6. 最后再决定是否值得长期维护

超越官方库常见，但只在下面几类场景最值得：

- 超高频热点路径
- shape 很稳定
- 上下游能一起融合
- 延迟或成本收益足够大

不值得的情况也很多：

- shape 非常动态
- 官方库已经高度特化
- 维护成本、兼容性、数值风险太高

一句话总结：

> 超越官方库的关键不是“写更底层”，而是“利用更强的 workload 先验，把通用问题改造成特化问题”。

## 6. 融合算子的设计思路是什么？

先给结论：

> 融合算子不是“把能拼的都拼起来”，而是围绕数据流做设计：哪些中间结果最贵、哪些数据一旦读上来就应该多用几次、哪些阶段应该留在寄存器/shared memory，哪些阶段必须落回 HBM。

### 1. 先找值得融合的链路

最值得融合的通常是这三类：

- 多个 pointwise / reduction 小算子串联，launch 多、访存重
- 中间结果很大，但生命周期极短
- 上下游天然共享同一批数据访问

经典例子：

- `bias + gelu`
- `dropout + residual + layernorm`
- `rmsnorm + residual`
- attention 中的 `score + mask + softmax + value aggregation`

### 2. 融合边界的核心不是“语义相邻”，而是“数据相邻”

判断两个操作能不能融合，最关键看：

- 是否共享输入 tile
- 中间结果是否可在寄存器或 shared memory 中表示
- 是否会引入过大的同步需求
- 是否会导致不可接受的寄存器压力

所以“数学上连续”不等于“实现上适合融合”。

### 3. 常见设计顺序

一个比较稳的思路是：

1. 先画出原始算子链的数据读写路径
2. 找到最大中间张量和重复读写点
3. 决定哪些阶段在片上完成
4. 再选择 kernel 形态：pointwise fusion、fused reduction、persistent kernel、tile-wise fusion

也就是说，先做内存账本，再写 kernel。

### 4. 融合时最常见的收益来源

- 少一次或多次 HBM round-trip
- 少几个 kernel launch
- 更好的 cache / SRAM 复用
- 更规则的线程映射
- 把多个低 intensity 小算子拼成更高 intensity 的大算子

这也是为什么融合常常和 [[../wiki/concepts/Tiling|Tiling]]、[[../wiki/concepts/重计算|重计算]]、[[../wiki/concepts/FlashAttention|FlashAttention]] 一起出现。

### 5. 融合不一定更快，最常见的反噬点

- 寄存器压力上升，导致 occupancy 下降
- shared memory 占用变大，限制并发 block 数
- 调试和数值验证复杂度提高
- 编译时间、二进制体积、适配成本上升
- 某些阶段被迫共享一个 launch 形态，反而损失局部最优

### 6. 一个实用判断标准

如果一个融合方案：

- 明显减少大中间张量读写
- 没把寄存器 / shared memory 压爆
- workload shape 足够稳定

那它通常值得做。

如果只是省了一次很小的 pointwise launch，却把 kernel 复杂度翻倍，往往不值。

一句话总结：

> 融合算子的设计核心是“把中间结果尽量留在片上，并让一次读取服务更多计算”，而不是机械地把多个 op 名字拼在一起。

## 7. 混合精度训练与推理涉及哪些关键问题？

先给结论：

> 混合精度的本质是：把“必须高精度”的部分和“可以低精度”的部分拆开处理，用更低的数据精度换内存、带宽和 Tensor Core 吞吐，但不能把数值稳定性一起丢掉。

### 1. 先区分几个常见精度角色

- `FP32`：高精度基线，常用于 master weights、部分累加和统计量
- `FP16`：吞吐高、显存省，但动态范围窄，训练时更容易下溢/上溢
- `BF16`：指数位和 `FP32` 一样宽，训练稳定性通常比 `FP16` 更好
- `FP8`：进一步压缩带宽和存储，但对 scale、校准和硬件支持更敏感

### 2. 训练里最关键的 6 个问题

#### 问题 1：哪些张量能低精度，哪些不能

通常会把：

- matmul / attention / activation 放在低精度
- master weights、部分 reduction、norm 统计量、softmax 关键归一化留在更高精度

原则不是“一律低精度”，而是“高吞吐路径低精度，数值脆弱路径高精度”。

#### 问题 2：梯度下溢和 loss scaling

`FP16` 的常见问题是小梯度变成 0，因此训练里常配合 `loss scaling`。

- 静态 scaling：实现简单，但需要调参
- 动态 scaling：更稳，会根据 overflow 自动调节

如果用 `BF16`，loss scaling 需求通常比 `FP16` 弱很多，但并不意味着所有数值问题都消失。

#### 问题 3：累加精度

即便输入是低精度，很多 matmul / reduction 也会：

- 低精度输入
- 更高精度 accumulate

这是混合精度能兼顾速度和稳定性的关键，不然误差会在长链路中快速放大。

#### 问题 4：优化器状态和更新路径

训练里常见做法是：

- 维护 `FP32 master weights`
- 梯度先反缩放
- 再做 clipping、weight decay、optimizer update

如果更新路径全放在低精度里，训练更容易漂。

#### 问题 5：分布式训练中的精度一致性

在多卡场景还要考虑：

- all-reduce 之前还是之后 cast
- 通信 dtype 用什么
- 不同 rank 的 scale 是否一致
- 梯度 overflow 怎么同步处理

否则单卡稳定，多卡也可能不稳。

#### 问题 6：数值排障方式

混合精度问题很少直接表现为“这里坏了”，更常见的是：

- loss 突然 NaN
- 某层梯度全 0
- 收敛变慢但不报错
- 推理结果轻微漂移

因此要有分层检查：

- 激活/梯度分布
- 是否出现 inf/nan
- 哪一层最早失稳
- 关闭 AMP 后是否恢复

### 3. 推理里最关键的 5 个问题

#### 问题 1：权重 dtype 和激活 dtype 的组合

推理要区分：

- 权重是不是 `FP16 / BF16 / FP8 / INT8 / INT4`
- 激活是不是同样低精度
- 累加是不是仍在更高精度

不要把“混合精度”和“量化”完全混为一谈，但两者常常会一起出现。

#### 问题 2：KV Cache 的精度

长上下文推理里，`KV Cache` 往往是大头之一，因此常见优化包括：

- `FP16 -> BF16`
- `FP16/BF16 -> FP8`
- 更激进的 cache quantization

但这会影响：

- 长上下文稳定性
- attention 误差累积
- 首 token 与长 decode 质量

#### 问题 3：dequant / cast 本身也有成本

低精度并不总是更快。如果：

- 频繁 cast
- dequant 额外 kernel 太多
- 没命中 Tensor Core 的高效路径

那可能显存省了，但吞吐没明显变好。

#### 问题 4：不同阶段的最优精度可能不同

在推理里经常看到：

- prefill 更像大矩阵计算，适合极致吃 Tensor Core
- decode 更像小 batch、memory-bound、强依赖 `KV Cache`

所以同一种精度策略在 prefill 和 decode 的收益并不对称。

#### 问题 5：精度切换要和 benchmark 一起看

只看困惑度或只看吞吐都不够，要同时看：

- 吞吐
- 首 token 延迟
- token/s
- 显存占用
- 长上下文质量

一句话总结：

> 混合精度不是一个“开关”，而是一组数值与系统共同设计问题：哪些地方降精度、哪些地方保精度、哪些地方高精度累加、哪些地方通信和缓存也要跟着改。

## 8. GPU 性能优化中，Nsight 工具链如何使用？应关注哪些计算效率与带宽效率指标？

先给结论：

> `Nsight Systems` 用来回答“时间花在哪、流水线哪里断了”，`Nsight Compute` 用来回答“单个 kernel 为什么慢”。实际工作流通常是先 `nsys` 再 `ncu`，而不是反过来。

### 1. 一条实用工作流

#### 第一步：先 benchmark，确认问题稳定存在

先把：

- warmup
- 多次重复
- 固定输入 shape
- 计时边界同步

都做对，不然 profiler 很容易追错对象。

#### 第二步：用 `nsys` 看全局时间线

`Nsight Systems` 适合看：

- CPU 是否及时 launch
- GPU timeline 是否存在大空洞
- memcpy、NCCL、compute 是否重叠
- 多 stream 是否真正并发
- prefill / decode / communication 各阶段占比

如果端到端瓶颈是：

- Python 调度
- CPU launch
- H2D / D2H 拷贝
- stream 同步
- NCCL 等待

那 `ncu` 往往帮不了太多。

常见命令：

```bash
nsys profile -t cuda,nvtx,osrt,cublas,cudnn -o trace python your_script.py
```

如果程序较长，最好用 `NVTX` 或 `cudaProfilerStart/Stop` 只截关键区间。

#### 第三步：锁定热点 kernel 后，再上 `ncu`

`Nsight Compute` 适合看：

- 这个 kernel 是 compute-bound 还是 memory-bound
- occupancy 是否受寄存器或 shared memory 限制
- warp stall 主要卡在哪
- shared memory / L1 / L2 / DRAM 行为
- Tensor Core 是否真的打满了高吞吐路径

常见命令：

```bash
ncu --set full --kernel-name regex:your_kernel python your_script.py
```

### 2. `nsys` 里最值得先看的信号

#### 时间线信号

- GPU 是否有长时间 idle gap
- CPU launch 和 GPU 执行之间是否断流
- memcpy 是否插进了热点路径
- 同步点是不是把并发压没了
- NCCL 和计算是否重叠

#### GPU metrics 信号

`Nsight Systems` 的 GPU metrics 很适合先看粗粒度趋势，重点关注：

- `SM Active`
- `SM Issue`
- `Tensor Active`
- `SM Warp Occupancy`
- `DRAM Bandwidth`
- `NVLink / PCIe Bandwidth`

这些名字会随 GPU 和版本略有差异，但问题意识基本一致：

- GPU 是不是闲着
- SM 发射率高不高
- Tensor Core 用没用起来
- 带宽是不是打满
- 是不是被互联拖住

### 3. `ncu` 里最值得看的两大类指标

注意：不同架构和 Nsight 版本下，具体 metric 名称会变，但关注点基本稳定。

#### A. 计算效率指标

优先看这些：

- achieved occupancy 和理论 occupancy 的差距
- `SM` 或各类 compute pipe 的 utilization
- `Tensor Core` 利用率
- instructions per cycle / issue rate
- warp stall reason
- branch divergence

这类指标回答的是：

- 算力单元有没有吃饱
- 线程是否有足够并发去隐藏延迟
- 是不是因为寄存器、shared memory、分支分歧导致发射效率低

#### B. 带宽效率指标

优先看这些：

- DRAM throughput 是否接近峰值
- L2 hit rate
- global memory 请求是否合并良好
- shared memory bank conflict
- bytes/FLOP 或者更高层的 arithmetic intensity 线索
- memory pipe busy、memory throttling、load/store stalls

这类指标回答的是：

- 你的 kernel 是不是主要卡在搬数据
- 数据是不是在 L2/L1/shared memory 层面复用得足够好
- 是不是因为 coalescing 差或 bank conflict 浪费了带宽

### 4. 用指标做判断时的 4 个常见套路

#### 套路 1：DRAM 很高、SM 不高

通常说明更偏 memory-bound。

先想：

- 融合
- tiling
- 数据复用
- coalescing
- cache 命中

#### 套路 2：occupancy 很低，但单线程资源很重

先不要机械追高 occupancy，要看是不是：

- 寄存器太多
- shared memory 太大
- block 太重

如果当前 kernel 已经 compute-heavy，适度低 occupancy 也可能是合理结果。

#### 套路 3：Tensor Active 很低

先检查：

- 是否命中 Tensor Core kernel
- 矩阵维度是否满足对齐偏好
- dtype 是否合适
- launch shape 是否导致小矩阵太碎

#### 套路 4：timeline 上 GPU 有大量 gap

这时往往先别优化 kernel，而是先解决：

- CPU launch 不及时
- 小 kernel 太碎
- 缺少 CUDA Graphs
- 不必要同步
- 数据传输位置不对

### 5. 面试里可以怎么总结

一个比较稳的答法是：

> 我会先用 benchmark 把问题稳定复现，再用 `Nsight Systems` 看端到端时间线，判断瓶颈在 CPU launch、数据传输、同步还是 GPU 计算本身。锁定热点 kernel 之后，再用 `Nsight Compute` 看 occupancy、SM/Tensor Core 利用率、warp stall、L2/DRAM 吞吐、coalescing 和 bank conflict，最后判断该优先做融合、tiling、layout 重排，还是调整 launch 与并发。

## 9. KV Cache 的工作原理是什么？FlashAttention 的内存优化核心思想是什么？

### 9.1 KV Cache 的工作原理

先给结论：

> `KV Cache` 的本质是把“历史 token 的 K/V 状态”持久化下来，让 decode 阶段不再重算整段上下文，而只为新 token 计算一次查询，再去读取历史缓存。

#### 没有 KV Cache 时

每生成一个新 token，都要把到当前长度为止的整段序列重新前向一遍。

这会导致 decode 阶段出现大量重复计算。

#### 有 KV Cache 时

流程会拆成两段：

1. `Prefill`
   - 对整段 prompt 并行算出各层 `K/V`
   - 把这些 `K/V` 写入 cache
2. `Decode`
   - 每次只为新 token 算一次新的 `Q/K/V`
   - 新 token 的 `Q` 去读历史 `KV Cache`
   - 新产生的 `K/V` 再追加写入 cache

所以 `KV Cache` 不是改变 attention 依赖关系，而是把历史状态保存下来，避免重复计算。

#### 一个常用的大小估算

如果忽略实现细节，`KV Cache` 大小通常近似与这些量线性相关：

- `batch size`
- `sequence length`
- `num_layers`
- `num_kv_heads`
- `head_dim`
- `dtype bytes`

可粗略记成：

`KV Cache bytes ≈ 2 * B * S * L * H_kv * D_head * bytes_per_elem`

其中前面的 `2` 是因为要同时存 `K` 和 `V`。

#### 为什么它加速了推理，却又引入了新瓶颈

因为它把问题从：

- “重复算历史”

转成了：

- “反复读历史 cache”

所以 decode 阶段常常会变得更偏 memory-bound，这也是为什么：

- cache 布局
- [[../wiki/concepts/PagedAttention|PagedAttention]]
- cache 精度
- prefix 共享

会成为推理系统的重点。

一句话总结：

> `KV Cache` 的价值在于用显存换计算，把 decode 的主要成本从“重算整段序列”变成“读取并追加历史状态”。

### 9.2 FlashAttention 的内存优化核心思想

先给结论：

> `FlashAttention` 的核心不是少做 FLOPs，而是少做 HBM 往返。它通过 tile-wise 计算、online softmax 和融合，把注意力的大中间矩阵尽量留在片上，而不是反复写回显存。

#### 标准 attention 的痛点

标准实现往往会显式或半显式地产生：

- `QK^T` 分数矩阵
- softmax 概率矩阵
- 与 `V` 相乘前后的中间结果

这些中间张量在长序列下非常大，HBM 读写代价高。

#### FlashAttention 的核心做法

1. 把 `Q/K/V` 分块
2. 固定一个 `Q` block
3. 让多个 `KV` block 依次流过
4. 对每个 `Q` 行维护 online softmax 所需的运行状态
5. 直接在片上累计输出，而不是物化完整概率矩阵

常见运行状态可以理解为：

- 行最大值
- 归一化因子
- 当前输出累计值

这样每处理完一个 `KV` tile，都只更新这几个小状态，而不是把整个中间 attention matrix 写回 HBM。

#### 为什么 online softmax 是关键

因为普通 safe softmax 通常需要完整看到一整行分数，才能：

- 找最大值
- 做指数
- 求和归一化

而 online softmax 允许你按 tile 逐步更新这些统计量，最后得到和整行计算等价的精确结果。

#### 为什么很多实现还会结合重计算

因为在 GPU 上：

- 多存一次大中间张量
- 和多算一点局部计算

相比之下，后者往往更便宜。

所以 `FlashAttention` 常会选择：

- 少存
- 多算一点

用重计算换更低的 IO 成本。

#### 为什么它在 prefill 更有价值

因为 prefill 阶段：

- 序列长
- 并行度高
- attention score matrix 很大

更容易被中间矩阵 IO 拖住。

decode 阶段通常更小、更碎、更偏 cache 读，因此收益结构和 prefill 不完全一样。

一句话总结：

> `FlashAttention` 的内存优化核心思想，就是把 exact attention 重写成“按块流式处理 + 在线归一化 + 片上累计输出”的 IO-aware 数据流。

## 一页速记

- 特定 shape 调优能赢官方库，靠的是更强先验和更深特化，不是盲目重写
- 算子融合的核心是减少 HBM round-trip，而不是机械拼 op
- 混合精度要同时管理 dtype、累加精度、loss scaling、优化器状态和 cache 精度
- `nsys` 看全局时间线，`ncu` 看单 kernel 微观瓶颈
- `KV Cache` 用显存换重复计算，把 decode 推向 memory-bound
- `FlashAttention` 用 tile + online softmax + 片上累计输出减少 HBM IO

## 参考

### Vault 内

- [[../wiki/sources/斯坦福CS336 Lecture 5 - GPUs|斯坦福CS336 Lecture 5 - GPUs]]
- [[../wiki/sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing|斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../wiki/sources/你一定要知道：CUDA优化六要|你一定要知道：CUDA优化六要]]
- [[../wiki/sources/LLM推理优化核心技术|LLM推理优化核心技术]]
- [[../wiki/sources/Flash Attention 详细解释推演与Pytorch代码实现|Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../wiki/sources/美团一面：请介绍 vLLM PageAttention|美团一面：请介绍 vLLM PageAttention]]

### 外部参考

- [NVIDIA Docs: Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [NVIDIA Docs: Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Docs: Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
