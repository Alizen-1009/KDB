# vLLM v0 与 vLLM v1 调度架构差异截图整理

## 来源说明

- 来源：用户提供的三张截图
- 类型：截图转写 / 面试知识点整理
- 整理日期：2026-05-07
- 备注：用户原问题中写作 `VLM V1 / V0`，截图内容实际指向 `vLLM v1 / v0`。本文按截图转写为 `vLLM`。

## 核对摘要

这组三张图的主线基本正确：`vLLM v0` 更像以 prefill/decode 阶段为中心的调度；`vLLM v1` 通过 `{request_id: num_tokens}` 这类 token-level scheduling decision 把 prompt token 与 output token 放进统一调度视图中，从而更自然地支持 chunked prefill、prefix caching 和 speculative decoding。

需要修正或标注的地方：

1. `vLLM v0` 不是绝对不能混合 prefill 和 decode；在开启 chunked prefill 后，v0 也可在 token budget 约束下把 decode 和部分 prefill 放入同一调度周期。截图更适合作为“默认/简化 v0 调度路径”的理解。
2. `vLLM v1` 的 `token quota` 更准确地说是“每个调度步为请求分配要处理的 token 数”，不是长期固定绑定在请求上的静态配额。公开调优项通常应关注 `max_num_batched_tokens`、`max_num_seqs`、scheduler policy 等，而不是把 `token_quota` 当作稳定公开参数。
3. 多 GPU 部分说“vLLM 主要支持数据并行”过窄。vLLM serving 可以通过副本路由实现请求级并行，也支持 tensor parallel、pipeline parallel 等模型并行形态；需要区分“服务层路由调度”和“单个请求的分布式执行”。

## 截图内容转写

### 三、调度决策流程：vLLM v0 vs vLLM v1

vLLM 的调度器在 v0 和 v1 版本中有本质区别。理解这两个版本的调度差异，才能知道 vLLM 的演进方向。

### vLLM v0：阶段分离调度器

v0 调度器严格区分 Prefill（预填充）和 Decode（解码）两个阶段：

#### 架构特点

- 双阶段分离：Prefill 和 Decode 被视为不同的计算模式
- 条件性 Chunked Prefill：长 prompt 可被分块，但需手动开启
- Swapping 机制：显存不足时，KV Cache 可换出到 CPU 内存

#### 调度决策流程（v0）

```python
# 伪代码：v0 调度逻辑（简化）
def schedule_step_v0():
    # 1. 分离处理 Prefill 和 Decode 请求
    prefill_requests = filter_prefill(waiting_queue)
    decode_requests = filter_decode(running_queue)

    # 2. 优先调度 Prefill（保证 TTFT）
    if prefill_requests:
        batch = select_prefill(prefill_requests)
        # 可能独占 GPU，阻塞 Decode
        execute_prefill(batch)

    # 3. 再调度 Decode
    if decode_requests:
        batch = select_decode(decode_requests)
        execute_decode(batch)

    # 4. 状态转移：Prefill 完成后转为 Decode
    move_finished_prefill_to_decode()
```

#### v0 流程图

```text
新请求到达 -> Waiting队列（标记为Prefill）
  ↓
调度周期开始
  ↓
1. 检查Swapped队列 -> 有空间则换入
  ↓
2. 优先选择Prefill请求 -> 保证首token延迟
  ↓
3. 分配Block资源 -> 检查prefix cache
  ↓
4. 组装Batch -> 可能独占GPU（长prefill）
  ↓
5. GPU执行 -> Prefill完成后转为Decode
  ↓
6. Decode请求组batch执行 -> Continuous Batching
  ↓
回到步骤1，下一个调度周期
```

#### v0 的问题

1. 阶段干扰：长 Prefill 会阻塞 Decode 请求
2. 复杂度高：需协调两个阶段的资源分配
3. 功能集成难：Chunked Prefill、Speculative Decoding 等需特殊处理

### vLLM v1：统一调度器（Unified Scheduler）

v1 调度器是 vLLM 架构重构的核心，采用 Token 统一调度模型：

#### 架构特点

- 统一视图：不再区分 Prefill 和 Decode，所有请求按 token 消耗统一管理
- 动态配额分配：每个请求分配固定的 token 配额（包括 prompt 处理和生成）
- 默认 Chunked Prefill：长 prompt 自动分块，无需特殊配置
- 移除 Swapping：简化架构，不再需要 KV Cache 换入换出

#### 调度决策流程（v1）

```python
# 伪代码：v1 调度逻辑（简化）
def schedule_step_v1():
    # 1. 统一视图：所有请求按 token 消耗管理
    requests = all_requests()  # 不区分 prefill/decode

    # 2. 为每个请求分配 token 配额
    quotas = assign_token_quotas(requests)
    # 示例：{request_id: 512}  # 每个请求本轮最多消耗512个token

    # 3. 根据配额组装统一 batch
    batch = []
    for req in requests:
        # 消耗配额处理 prompt 剩余部分或生成新 token
        tokens_to_process = min(quotas[req.id], req.remaining_tokens)
        batch.append((req, tokens_to_process))

    # 4. 统一执行（包含 prompt chunking 和 generation）
    execute_unified_batch(batch)

    # 5. 更新剩余 token 数，请求完成则移除
    update_remaining_tokens()
```

#### v1 流程图

```text
新请求到达 -> Waiting队列（统一视图）
  ↓
调度周期开始
  ↓
1. 为每个请求分配token配额 -> 动态计算
  ↓
2. 根据配额选择处理的token数 -> 包含prompt和生成
  ↓
3. 组装统一batch -> 自动应用Chunked Prefill
  ↓
4. GPU统一执行 -> 不区分prefill/decode阶段
  ↓
5. 更新配额和剩余token数 -> 请求完成则移除
  ↓
回到步骤1，下一个调度周期
```

### 核心对比

| 特性 | vLLM v0 | vLLM v1 |
| --- | --- | --- |
| 调度模型 | 阶段分离（Prefill/Decode） | 统一调度（Token 级别） |
| Chunked Prefill | 有条件启用，需配置 | 默认启用，自动处理 |
| 资源管理 | 复杂（需协调两个阶段） | 简化（统一 token 配额） |
| Swapping | 支持（显存不足时） | 移除（简化架构） |
| 功能集成 | 困难（需适配两个阶段） | 容易（统一接口） |
| 调度策略 | FCFS（默认） | FCFS + 优先级调度 |

### 演进意义

1. 性能提升：v1 在长上下文场景中有显著性能提升（官方数据）
2. 简化架构：统一调度器减少系统复杂性和技术债
3. 未来友好：为 Speculative Decoding、动态批处理等高级功能提供更好基础

实践建议：

- 新项目直接用 v1：统一调度器更简单，性能更好
- 迁移注意事项：v1 移除 Swapping，需确保显存足够
- 调优重点：v1 关注 `token_quota` 参数（控制每个请求每步的处理量）

## 四、调度优化技巧

### 1. 延迟与吞吐的权衡

- 偏向吞吐：让 batch 尽量大，GPU 利用率高，但单个请求延迟可能增加
- 偏向延迟：batch 小，响应快，但 GPU 可能没喂饱

实践建议：设置 `max_batch_size` 和 `min_batch_size`，根据 SLA 调整。

### 2. 长尾延迟治理

长尾延迟（P99延迟）往往由少数长请求或资源竞争引起。

应对策略：

- 隔离队列：长短请求分队列，避免互相影响
- 抢占式调度：短请求可抢占长请求的资源（复杂，vLLM 暂未实现）
- 限流：拒绝明显超长的请求

### 3. 多 GPU 调度

多卡时调度更复杂：

1. 数据并行：每个 GPU 有完整的模型副本，请求可以路由到任意卡
2. 模型并行：模型切分到多卡，请求必须跨卡执行

vLLM 主要支持数据并行，调度器需要：

- 监控各 GPU 负载
- 均衡分配请求
- 处理 GPU 间通信

## LLM 核对备注

- `vLLM v1` 官方博客明确把 scheduler decision 表示为 `{request_id: num_tokens}`，即每一步决定每个请求处理多少 token；这支持截图里“统一 token 调度”的主线。
- `vLLM v1` 去掉了传统 prefill/decode 两阶段在调度表示上的强区分，但内核执行和请求状态仍然会区分 prompt 剩余 token、decode token、KV block 分配等工程细节。面试回答里应避免说成“prefill 和 decode 完全不存在”。
- `vLLM v1` 迁移时确实要注意 swapping 能力变化。更稳妥的表述是：v1 不再沿用 v0 的 CPU swap 机制，资源不足时更依赖 KV cache capacity、admission control、recompute/preemption 等机制或服务层限流。
- `max_batch_size / min_batch_size` 是通用服务调优说法；对于 vLLM，常见公开参数更应落到 `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`gpu_memory_utilization`、scheduler policy 等具体配置。

## 核对参考

- vLLM V1 alpha release blog: https://vllm.ai/blog/v1-alpha-release
- vLLM V1 usage guide: https://docs.vllm.ai/en/stable/usage/v1_guide.html
- vLLM optimization and tuning docs: https://docs.vllm.ai/en/latest/configuration/optimization.html
- vLLM scheduler config docs: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
