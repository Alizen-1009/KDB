# 斯坦福CS336 Lecture 7 - Parallelism basics

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方课程仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方讲义 PDF：https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：课程讲义

## 原始说明

- 这一讲是 Stanford CS336 从单卡性能工程过渡到多卡多机训练系统的关键节点。
- 讲义分三部分：网络通信基础、标准 LLM 并行原语、以及大规模训练时如何组合这些并行方式。
- 主题重点不是某个框架 API，而是并行训练的系统权衡：内存、带宽、batch size、利用率和实现复杂度。

## 讲义结构

### Part 1: Networking basics for LLMs

- 先说明为什么单 GPU 在 compute 和 memory 上都很快触顶。
- 引入多机多卡训练时最重要的 collective communication 原语：
  - all-reduce
  - broadcast
  - reduce
  - all-gather
  - reduce-scatter
- 讲义特别强调：在带宽受限条件下，`all-reduce ≈ reduce-scatter + all-gather` 是非常关键的等价视角。

### Part 2: Standard LLM parallelization primitives

- 先从最朴素的数据并行出发，指出它的三种尺度问题：
  - 内存不扩展
  - 通信每步都要同步梯度
  - batch size 不能无限扩大
- 然后引入 ZeRO：
  - Stage 1：shard optimizer state
  - Stage 2：进一步 shard gradients
  - Stage 3：连参数也一起 shard，讲义把它和 FSDP 联系起来
- Beyond data parallel 之后，转向 model parallel：
  - Pipeline parallel：沿层深切
  - Tensor parallel：沿层宽切
  - Sequence parallel：把部分 activation memory 沿序列维切分

### Part 3: Putting it together

- 讲义反复强调：没有单一并行方案能同时解决所有问题。
- 实战上更像是组合题：
  - 先用 tensor parallel 吃掉节点内高速互联
  - 再用 pipeline parallel 扩到跨节点
  - 剩余部分用 data parallel 扩吞吐
- 当 activation 成为主要内存瓶颈时，sequence parallel 和 recomputation 变得关键。
- 讲义最后给出一种接近“3D parallelism”的经验法则：`TP + PP + DP` 联合使用，而不是押宝某一种并行。

## 从讲义中抽出的高信号结论

- ZeRO/FSDP 的核心不是“更神秘的数据并行”，而是用 shard 和 `reduce-scatter / all-gather` 重新组织状态与同步。
- Pipeline parallel 更像是在慢网络上用激活通信换参数通信，Tensor parallel 则是在快网络上用更高带宽消耗换掉 bubble。
- 真正的大模型训练并行不是“选哪一种”，而是根据内存、带宽、batch size 和网络拓扑去组合多种并行原语。
