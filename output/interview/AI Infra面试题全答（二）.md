# AI Infra面试题全答（二）

## 说明

这份文档补的是另一组更偏 `AI Infra 平台 / 参数服务器 / 调度 / 存储 / MLOps / 管理协作 / 行为面` 的题。  
风格仍然遵循：

- 先给结论
- 再讲机制
- 最后补 tradeoff、边界和工程判断

如果题目明显要求“结合个人经历”，我会给一版可替换模板，你再填自己的真实项目。

## 一面

### 1. 请详细介绍一个你参与过的 AI Infra 或大规模分布式系统项目，并说明你在其中的角色和贡献

这题不要按时间线流水账讲，最稳的结构是：

1. 项目目标
2. 系统规模
3. 你的职责边界
4. 最关键的技术难题
5. 你的方案
6. 可量化结果

模板：

> 我参与过一个 `训练/推理/平台` 系统项目，目标是解决 `吞吐不足 / 延迟过高 / 训练不稳定 / 资源利用率低` 的问题。当时系统规模大概是 `X` 张 GPU、`Y` 个节点，服务的模型规模是 `Z`。我负责的部分主要是 `分布式训练链路 / 调度 / KV cache / 性能分析 / 容错恢复`。当时最大的技术难题是 `A`，因为它会导致 `B`。我最后设计并推动落地了 `C`，包括 `具体机制 1/2/3`。最终效果是 `step time 降低 xx% / GPU 利用率提升 xx% / P99 降低 xx% / 成本下降 xx%`。这个项目让我比较完整地覆盖了从底层性能到系统稳定性的链路。

### 2. 在 AI 训练场景下，如何设计和实现一个高性能的参数服务器（Parameter Server）？需要考虑哪些关键点？

先给结论：

> 高性能参数服务器的核心不是“把参数放在一台机器上供大家拉取”，而是围绕参数分片、通信拓扑、热点削峰、一致性语义和容错机制，把更新路径做成低延迟、高吞吐、可扩展的分布式存储与同步系统。

关键设计点：

#### 1. 参数分片

- 按层、按张量范围、按 key range 做 shard
- 避免热点参数都落在同一 shard
- 训练推荐系统时 embedding 表往往天然按 id range 切分

#### 2. 更新语义

- 同步 PS：更稳定，但慢
- 异步 PS：吞吐高，但有 stale gradient
- SSP / bounded staleness：折中

#### 3. Push/Pull 路径

- worker 拉参数、推梯度
- server 聚合再更新
- 大系统里通常要做梯度压缩、稀疏更新、局部累积和分层聚合

#### 4. 网络和拓扑

- 节点放置要 topology-aware
- 可以做 server colocate，减少跨交换机流量
- 大 embedding/稀疏训练更依赖网络和缓存设计

#### 5. 缓存和热点

- 频繁访问参数可放本地 cache
- 热 key 做 replication
- 稀疏参数更新可用 batched pull/push 减少小包

#### 6. 容错

- shard 副本
- WAL / checkpoint
- worker 重试与幂等更新

#### 7. 为什么今天 PS 没有 all-reduce 主流

- dense LLM 训练更常用 all-reduce / ZeRO / FSDP
- PS 更适合超大稀疏参数场景，比如推荐系统 embedding

一句话：

> 参数服务器在 AI 训练里依然重要，但它更偏向稀疏、超大 embedding、非严格同步场景；dense 大模型训练更常见的是 collective-based 路线。

### 3. 谈谈对 CUDA 编程和 GPU 显存管理的理解，如何避免显存溢出（OOM）和优化 kernel 性能？

我会把它拆成两部分回答。

#### 1. CUDA 编程理解

CUDA 的本质是：

- 用 `grid / block / thread` 把单线程逻辑映射到大规模并行执行
- 性能关键不只在计算量，更在访存模式、shared memory、occupancy、warp divergence、launch 配置

#### 2. 显存管理理解

显存主要消耗在：

- 参数
- 梯度
- optimizer states
- 激活
- KV cache
- 临时 workspace

避免 OOM 的手段：

- BF16/FP16/FP8
- activation checkpoint
- ZeRO / FSDP / offload
- 减小 batch / sequence length
- 更好的 KV cache 管理
- 避免张量生命周期过长
- 尽量减少不必要的 `torch.cuda.empty_cache()` 误用和显存碎片

优化 kernel 性能的抓手：

- 先看 memory-bound 还是 compute-bound
- coalesced access
- shared memory tiling
- 避免 bank conflict
- 减少 warp divergence
- 选合适 block size
- 尽可能 fusion / CUDA Graphs / Triton / FlashAttention 风格重排数据流

一句话：

> 避免 OOM 是做内存账本，优化 kernel 是做数据流账本；两件事本质上都在回答“哪些数据必须留在 GPU，哪些访问路径是最贵的”。

### 4. 在分布式深度学习训练中，数据并行、模型并行、流水线并行的区别是什么？各自适用于什么场景？

#### 数据并行 DP

- 切 batch
- 每卡一份完整模型
- 适合模型能放进单卡、目标是放大吞吐

#### 模型并行 MP

这里通常面试官既可能指 TP，也可能泛指模型切分。

- 切模型本身
- 适合单卡放不下的大模型

#### 流水线并行 PP

- 按层深切 stage
- 适合超大模型跨设备存放
- 对 micro-batch 和 bubble 很敏感

一句话比较：

- `DP`：最简单，扩吞吐
- `TP/MP`：层内切分，压单卡压力和时延
- `PP`：层间切分，扩模型规模

### 5. 场景题：假设训练任务出现通信瓶颈（如 AllReduce 耗时过长），你会从哪些维度进行排查和优化？

按五层排查：

#### 1. 通信模式

- 是 `all_reduce` 过多，还是张量太碎
- bucket 是否过小
- 是否缺少通信与计算 overlap

#### 2. 拓扑

- 单机内还是跨节点
- NVLink / NVSwitch / IB 是否跑满
- rank 映射是否合理

#### 3. 程序实现

- DDP bucket 配置
- 是否有梯度同步过早触发
- 是否某些参数没有 bucket 化导致小通信包过多

#### 4. 训练配置

- batch 太小导致计算压不住通信
- TP/PP/DP 组合不合理
- 梯度累积是否能降低同步频率

#### 5. 系统手段

- 压缩梯度
- reduce-scatter 替换一部分 all-reduce
- 调整 placement
- 用更 topology-aware 的并行切分

一句话：

> 通信瓶颈不能只看 NCCL 耗时，要判断是“通信量太大”“通信太碎”“通信拓扑错了”，还是“计算太少压不住通信”。

### 6. 如何实现一个高效的分布式训练任务调度和资源管理系统？与传统的批处理调度（如 YARN, K8S）有何不同？

先给结论：

> AI 训练调度系统和传统批处理调度最大的区别，在于它调度的不是一堆松散 CPU 任务，而是强 gang、强拓扑、强容错、强环境一致性的 GPU 分布式作业。

关键能力：

- gang scheduling：所有 worker 一起就绪才开跑
- topology-aware placement：尽量把强通信任务放近
- 配置一致性：CUDA/NCCL/driver/image 对齐
- checkpoint / resume
- 优先级 / 抢占 / 配额
- 监控与自动失败恢复

与传统 YARN/K8S 的差异：

- AI 作业对网络拓扑更敏感
- 资源不是纯 CPU/内存，而是 GPU、HBM、IB、NVLink
- 失败恢复成本更高
- gang 和同步语义更强

### 7. 了解主流深度学习框架（PyTorch/TensorFlow）的分布式训练通信后端（如 NCCL，Gloo）吗？它们的优劣是什么？

#### NCCL

- 强项：GPU collective 性能强
- 适合：多 GPU 训练
- 弱点：更依赖 CUDA 和硬件环境，debug 门槛相对高

#### Gloo

- 强项：CPU 环境通用，部署简单
- 适合：CPU 分布式、控制面、小规模测试
- 弱点：GPU collective 性能不如 NCCL

#### TensorFlow 生态

- 常见会配合 NCCL、XLA、grpc、collective ops
- 更强调图级编译和策略抽象

一句话：

> GPU 训练后端首选 NCCL，Gloo 更像通用和兜底后端；真正选型主要看硬件、规模和要不要极致 GPU 通信性能。

### 8. AI 训练中 Checkpoint 的保存与恢复如何设计，以保证训练容错和高可用？

一个完整 checkpoint 至少包括：

- model weights
- optimizer states
- scheduler state
- dataloader / sampler progress
- 随机数状态
- 全局 step / epoch

设计要点：

#### 1. 一致性

- 保存时要确保各 rank 状态一致
- 最好有 barrier 或 coordinator

#### 2. 增量和分片

- 大模型 checkpoint 要分片
- 可以异步落盘
- 必要时增量 checkpoint

#### 3. 存储层

- 本地 NVMe 做 staging
- 再异步刷对象存储 / 分布式文件系统

#### 4. 恢复语义

- 支持从最近成功点恢复
- 支持 partial failure restart

一句话：

> Checkpoint 的难点不只是“存下来”，而是如何在大规模分布式环境里，用可接受的开销保存一致、可恢复、可验证的训练状态。

### 9. 如何对训练任务进行性能 Profiling？通常会关注哪些指标（如 GPU 利用率、TFLOPS）？

我一般分三层：

#### 1. 作业级

- step time
- samples/s 或 tokens/s
- global batch
- loss 是否稳定

#### 2. 设备级

- GPU util
- SM util
- HBM bandwidth
- achieved TFLOPS
- 显存占用

#### 3. 通信级

- all-reduce time
- overlap ratio
- link bandwidth
- NCCL error / timeout

工具：

- Nsight Systems
- Nsight Compute
- PyTorch Profiler
- nvidia-smi / DCGM

### 10. 谈谈对 AI 编译栈（如 XLA，TVM）的理解，它们是如何优化计算图的？

AI 编译栈的核心作用是：

- 把高层计算图降到更适合硬件执行的低层表示
- 做融合、重排、常量折叠、layout 优化、算子选择和代码生成

#### XLA

- 更偏框架深度整合
- 擅长图级优化和后端 lowering

#### TVM

- 更偏可编程编译栈
- 强在 schedule、自动调优和跨硬件后端

一句话：

> AI 编译栈的目标不是改变模型语义，而是把同一个计算图重新组织成更适合具体硬件的数据流和执行计划。

### 11. 如何监控和管理大规模 GPU 集群的健康状态与故障？

要监控：

- GPU：温度、功耗、ECC、Xid、util
- 网络：IB error、重传、吞吐
- 作业：失败率、重试率、checkpoint 成功率
- 节点：宕机、掉卡、驱动异常

还要有：

- 告警分级
- 自动隔离坏节点
- 作业重调度
- 故障归因 dashboard

### 12. 手写算法：实现一个线程安全的 LRU 缓存

```python
from collections import OrderedDict
from threading import Lock


class ThreadSafeLRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = OrderedDict()
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
            self.data[key] = value
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)
```

## 二面

### 1. 自我介绍

模板：

> 我主要聚焦在 AI Infra 和大模型系统方向，比较熟悉两条链路：一条是分布式训练，包括并行策略、通信和显存优化；另一条是推理系统，包括 KV cache、调度、低时延和高吞吐优化。我做事情时更偏系统视角，会把问题拆成硬件、kernel、运行时和集群四层来定位瓶颈和做权衡。

### 2. 深入聊聊你简历上提到的一个最具挑战性的 AI 系统项目。当时面临的核心技术难题是什么？你是如何设计解决方案并推动落地的？

模板：

> 最具挑战性的项目是 `X`。当时核心难题不是单点性能，而是 `性能、稳定性、成本` 三者一起受限。比如系统会在 `高并发/大模型/长上下文/多机训练` 下暴露出 `通信瓶颈、显存瓶颈、失败恢复慢` 的问题。我当时做的第一件事不是直接改代码，而是先建立指标闭环，把问题拆成 `作业级、设备级、通信级`。在此基础上我设计了 `A/B/C` 三个关键机制，并推动和算法、平台、业务一起灰度上线。最后效果是 `xx`。这个项目对我最大的锻炼是，很多系统问题最后不是靠一个优化点解决，而是靠机制设计、观测和跨团队协作一起收敛。

### 3. 项目深究：在模型推理服务（Serving）中，如何实现高吞吐、低延迟？有哪些模型优化和工程优化手段？

分四层答：

#### 模型层

- 量化
- 剪枝
- 蒸馏
- MoE / MQA / GQA / 更友好的架构

#### kernel 层

- FlashAttention / FlashDecoding
- fused kernels
- CUDA Graphs
- TensorRT / Triton

#### 运行时层

- KV cache
- PagedAttention
- Continuous batching
- Prefix caching
- Speculative decoding

#### 系统层

- 负载均衡
- cache-aware routing
- PD 分离
- 拓扑和 NIC 调优

### 4. 设计一个支持多模型、多版本、动态更新的在线推理服务平台，需要考虑哪些架构模块？如何做流量调度和灰度发布？

核心模块：

- 模型仓库与版本管理
- 部署控制面
- 推理执行引擎
- 流量网关
- 监控与效果回传

灰度发布方式：

- 按流量百分比
- 按用户桶
- 按地域/业务线
- canary + 自动回滚

流量调度要看：

- 模型版本
- 实例健康
- cache 命中
- GPU 负载

### 5. 如何保证分布式训练任务在数千张 GPU 卡上的稳定性和可扩展性？当训练任务频繁失败时，如何快速定位根因？

稳定性来自四件事：

- 标准化环境
- 强观测性
- 容错恢复
- 小步灰度扩容

快速定位根因要分层：

- 启动失败
- 数据问题
- 通信问题
- 硬件故障
- 程序逻辑问题

### 6. 在异构计算集群（包含多种型号的 GPU、AI 加速卡）环境下，如何设计资源调度策略以实现最优的资源利用率和任务性能？

关键是不要只按“卡数量”调度，而要按：

- 设备类型
- 显存大小
- 带宽和拓扑
- kernel 支持情况
- 作业画像

做法：

- 建资源池分层
- 做 capability-aware scheduling
- 让高要求作业绑定高端卡
- 低优先级或轻量作业消化碎片资源

### 7. 谈谈对大规模稀疏模型训练（如推荐系统）中 Embedding 层存储和通信优化的理解

Embedding 优化重点：

- 分片存储
- 热冷分层
- 稀疏更新
- batched lookup
- 参数服务器或 embedding cache
- 压缩通信

它和 dense LLM 最大不同是：

- 主要瓶颈常在 embedding lookup 和网络
- 参数巨大但每次访问稀疏

### 8. 中间件：在 AI Infra 中，消息队列（如 Kafka/Pulsar）可以用于哪些场景？与训练/推理流水线如何结合？

应用场景：

- 训练数据流转
- feature / sample pipeline
- 推理日志异步回传
- 模型效果埋点
- 训练/评估/部署事件流

和训练/推理结合方式：

- 推理服务异步把请求和结果写 MQ
- 训练侧消费做样本回流
- 评估系统消费做指标计算

### 9. 中间件：如何为 AI 场景设计和优化存储系统？例如，用于海量训练数据、特征、模型参数的存储，与传统对象存储有什么不同？

AI 存储更强调：

- 高吞吐顺序读
- 大量小文件治理
- 元数据效率
- 多层缓存
- checkpoint / dataset / feature store 多 workload 混合

和传统对象存储不同：

- AI workload 更容易同时要求高带宽和低元数据开销
- 训练作业对吞吐抖动更敏感

## 平台 / 管理 / 视野题

### 1. 从你的经验看，一个优秀的 AI Infra 团队应该具备哪些技术栈和能力？如何规划其技术演进路线？

优秀团队至少要覆盖：

- 训练系统
- 推理系统
- 集群平台
- 可观测性
- 编译和 kernel
- 存储与数据

演进路线常见是：

1. 先把作业跑稳
2. 再把吞吐和时延做起来
3. 再平台化、自动化
4. 再做更前沿的编译和架构优化

### 2. 如果由你从零开始搭建快手的 AI 基础设施平台，你会如何划分阶段和设定各阶段的里程碑目标？

#### 第一阶段：可用

- 训练任务能稳定跑
- 基本调度、镜像、监控、checkpoint 打通

#### 第二阶段：高效

- 多卡训练标准化
- 推理服务平台化
- GPU 利用率和成本显著改善

#### 第三阶段：平台化

- 自助化作业提交
- 自动扩缩容
- 多模型统一治理

#### 第四阶段：前沿能力

- 编译优化
- PD 分离
- 自动调优
- 更强 MLOps

### 3. 在 AI Infra 领域，如何建立有效的技术评估体系来衡量一个系统或组件的优劣（例如，对比两种参数同步策略）？

评估体系不能只看单指标，要同时看：

- 性能：吞吐、时延、加速比
- 成本：GPU 小时、带宽占用、功耗
- 稳定性：失败率、重试率、恢复时间
- 工程性：接入复杂度、维护成本、可观测性

### 4. 管理/协作：当你需要推动一个对业务方有短期成本增加但长期有利的基础设施项目时，你会如何与业务方沟通并获取支持？

核心是把“长期价值”翻译成“可量化收益 + 可控风险”。

做法：

- 先讲现状成本
- 再讲不做的代价
- 给分阶段收益
- 承诺灰度与回滚方案

### 5. 管理/协作：如何培养和提升 AI Infra 团队成员的技术视野和工程能力？你会鼓励他们关注哪些方向？

我会鼓励关注：

- 分布式训练
- 推理系统
- GPU kernel / 编译
- 网络和存储
- 可观测性与平台工程

培养方式：

- 读论文 + 读代码
- 做小型复现
- 技术分享
- postmortem 沉淀

### 6. 如何看待当前“大模型”趋势对 AI Infra 提出的新挑战？你认为未来的技术突破口会在哪里？

新挑战：

- HBM 和 KV cache 压力
- 网络瓶颈
- 功耗和机房约束
- 训练推理一体化效率

未来突破口：

- 更高效注意力与状态管理
- 更强量化
- 编译和自动调优
- 更智能的调度与缓存系统

### 7. AI 相关：在工程实践中，如何与算法研究员高效协作，将他们的模型想法快速、稳定地落地到生产系统？

关键是建立一条：

- 研究原型
- 最小系统实现
- 指标验证
- 生产灰度

的流水线。

我会把合作重点放在：

- 对齐目标函数
- 明确接口边界
- 先做最小可运行版本
- 再做性能化和平台化

### 8. AI 相关：如何设计一套系统或流程，来自动化地进行模型训练、评估、部署和效果监控（MLOps）？

至少要有：

- 数据和特征管理
- 训练 pipeline
- 评估 pipeline
- 模型 registry
- 部署控制面
- 在线指标与回流

一句话：

> MLOps 的本质是把模型生命周期做成可复现、可审计、可自动化迭代的闭环。

### 9. 在技术选型中，如何在追求前沿技术的创新性与保障系统的成熟稳定性之间做权衡？请举例说明。

模板：

> 我通常看三件事：收益是否足够大、风险是否可观测、回滚路径是否清晰。比如一个新推理引擎如果能带来 30% 吞吐提升，我会愿意做灰度试点；但如果收益只有 5%，却要重构整条 serving runtime，我会更保守，先通过小规模实验或离线 benchmark 验证。

### 10. 你最近在学习和研究哪些 AI Infra 相关的新技术或开源项目？你的学习方法和信息获取渠道是什么？

模板：

> 我最近在看 `PagedAttention / PD 分离 / Speculative Decoding / FP8 / FlashDecoding` 这些更贴近训练和推理系统交界的方向。我的方法一般是先看论文和官方文档，建立问题地图，再读核心代码或最小 demo，最后尽量自己写一份复盘或者做一个小实验。信息来源主要是论文、官方文档、开源仓库、社区 issue、以及业内技术分享。

### 11. 你如何保持自己在这个快速变化领域的技术敏锐度和竞争力？有长期的学习规划吗？

模板：

> 我会把学习分成“主线”和“支线”。主线是持续跟进训练、推理、编译和集群这几个核心方向；支线是每季度选一两个新热点做深入，比如最近的 speculative decoding 或 FP8。对我来说，真正有效的学习不是收藏资料，而是把知识写成结构化笔记，或者落成一个小实验和可复述的讲解。

### 12. 手写算法：设计一个支持动态扩容缩容的分布式一致性哈希算法

```python
import bisect
import hashlib


class ConsistentHashRing:
    def __init__(self, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = []
        self.node_map = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str):
        for i in range(self.virtual_nodes):
            vnode = f"{node_id}#{i}"
            h = self._hash(vnode)
            bisect.insort(self.ring, h)
            self.node_map[h] = node_id

    def remove_node(self, node_id: str):
        to_remove = []
        for h, nid in self.node_map.items():
            if nid == node_id:
                to_remove.append(h)
        for h in to_remove:
            idx = bisect.bisect_left(self.ring, h)
            if idx < len(self.ring) and self.ring[idx] == h:
                self.ring.pop(idx)
            del self.node_map[h]

    def get_node(self, key: str):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]
```

这个版本体现了：

- 一致性哈希环
- 虚拟节点
- 动态加节点
- 动态删节点

如果面试官继续追：

- 如何做副本：顺时针取多个不同物理节点
- 如何做负载均衡：不同节点不同 vnode 数量
- 如何做迁移最小化：一致性哈希天然保证只迁移受影响区间

## 行为面

### 1. 为什么选择快手，以及为什么对这个 AI Infra 岗位感兴趣？

模板：

> 我看重快手的原因主要有两点。第一，快手的业务规模和内容生态决定了它对 AI Infra 的要求不是停留在实验室，而是真正要面对大规模训练、推理、推荐、多模态和线上系统稳定性，这和我希望做的方向很一致。第二，我对 AI Infra 岗位感兴趣，是因为它处在模型能力和产品价值之间的关键位置：好的基础设施不仅能把成本打下来，也能把新模型能力更快、更稳地变成业务价值。

### 2. 回顾你的职业生涯，你认为自己最大的成就是什么？过程中遇到的最大困难是什么？

模板：

> 我觉得自己最大的成就是在 `X` 项目里，把一个原本 `不稳定 / 不高效 / 不成体系` 的系统推进到了 `可稳定上线 / 性能提升明显 / 被团队复用` 的状态。最大的困难通常不是某个技术点本身，而是要同时平衡性能、稳定性和跨团队协作，比如要在没有完美信息的情况下快速做方案取舍，并推动大家一起执行。

### 3. 你未来的职业发展规划是怎样的？希望在未来 3-5 年达到什么目标？

模板：

> 我希望未来 3-5 年继续深耕 AI Infra，既把训练和推理系统的深度做出来，也把平台和团队协作能力补齐。短期我希望在一个高标准环境里做更大规模、更复杂的系统；中期我希望能独立负责一条关键基础设施方向，从技术方案到落地、到团队协同都能扛起来。

### 4. 你通常如何应对工作压力或项目中的高挑战时刻？

模板：

> 我面对高压时通常不会先想着“硬扛”，而是先把问题拆小：目标是什么，最关键的风险是什么，哪些事情必须先做，哪些可以延后。对系统项目来说，很多压力来自不确定性，所以我会尽快建立事实基础，比如补指标、做最小复现、拉齐依赖方。这样压力就会从模糊变成可处理。

### 5. 你期望在一个什么样的团队和文化中工作？

模板：

> 我希望团队有几个特点：技术标准高、沟通直接、愿意复盘、对基础设施的长期价值有耐心。对我来说，一个好的团队不一定是“永远轻松”，而是大家能围绕同一个目标，把复杂问题说清楚、做扎实、沉淀下来。

### 6. 你如何看待工作与生活的平衡？

模板：

> 我认同阶段性冲刺在基础设施项目里是正常的，但我不认为长期高压是高质量工作的前提。真正好的平衡不是机械地算工时，而是团队能有清晰优先级、可预期节奏和合理复盘，这样大家在关键时期可以全力投入，平时也能保持持续学习和稳定输出。
