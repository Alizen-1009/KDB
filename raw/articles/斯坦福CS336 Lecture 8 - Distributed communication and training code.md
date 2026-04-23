# 斯坦福CS336 Lecture 8 - Distributed communication and training code

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方课程仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方脚本路径：https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_08.py
- 辅助文件：
  - `lecture_08_utils.py`
  - `lecture_08_remote_execute.sh`
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：可执行课程讲稿 / 代码

## 原始说明

- 这一讲是 Stanford CS336 对 Lecture 7 的直接代码化延伸：不再只是讲并行原理，而是用 `torch.distributed` 和多进程示例把这些原理真正跑起来。
- 讲稿围绕三类内容展开：
  - collective communication 的编程接口
  - `torch.distributed` / NCCL 的工程位置
  - bare-bones 的 data parallel / tensor parallel / pipeline parallel 代码骨架
- 重点不是生产级完备实现，而是把“最少代码下并行训练到底在做什么”讲清楚。

## 讲义结构

### 1. Distributed communication building blocks

- 讲稿先从编程接口层回顾 `broadcast / scatter / gather / reduce / all-gather / reduce-scatter / all-reduce`。
- 通过 rank / world size 的基本概念，把多进程分布式计算的心智模型固定下来。
- 讲稿随后直接用代码验证 `all-reduce = reduce-scatter + all-gather`。

### 2. torch.distributed 和 NCCL

- `torch.distributed` 被定位为 PyTorch 提供的高层通信接口。
- NCCL 被定位为把 collective operation 翻译成 GPU 间实际数据传输与 kernel 启动的底层库。
- 讲稿强调：真正的性能取决于硬件拓扑，比如 NVLink、NVSwitch、PCIe 和跨节点网络。

### 3. Communication benchmarking

- 讲稿直接 benchmark `all_reduce` 和 `reduce_scatter`，测通信耗时和“有效带宽”。
- 重点不是绝对数字，而是训练一种习惯：分布式系统优化同样需要 benchmark，而不是只看理论带宽。

### 4. Bare-bones distributed training patterns

- Data parallel：
  - 每个 rank 持有一份完整参数
  - 数据按 batch 维切分
  - backward 后对梯度 `all_reduce`
- Tensor parallel：
  - 每个 rank 持有每层的一部分参数
  - 前向时做局部 matmul，再通过 `all_gather` 拼回完整激活
- Pipeline parallel：
  - 每个 rank 持有一段层
  - 激活通过 `send / recv` 在 stage 间流动
  - 通过 micro-batch 减少 bubble

## 从讲稿中抽出的高信号结论

- Lecture 7 给的是并行训练的系统账本，Lecture 8 给的是“这些账本在代码里具体体现成什么通信调用”。
- `torch.distributed` 不是某个抽象黑盒，它只是把 collective 和 backend 组织成更可用的接口；真正的数据移动仍然要尊重硬件拓扑。
- 并行训练的本质差别，在代码层面经常就体现为：切分了什么、同步了什么、以及在哪个时刻通信。
