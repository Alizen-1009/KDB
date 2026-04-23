# Shared KV Cache

## 定义

在模型内部让最后若干层共享同一套 K/V 表示，而不是为每一层单独维护完整 KV Cache 的机制。

## 它解决什么问题

- 降低长上下文推理时的 KV Cache 内存占用
- 减少某些层重复进行 K/V 投影的计算成本

## 核心机制

- 前面的非共享层正常计算并存储 K/V
- 后面的共享层直接复用特定非共享层的 K/V
- 通过层间复用而不是请求间复用来压缩缓存成本

## 关键权衡

- 能减少缓存占用和部分投影计算
- 可能牺牲部分层级表达独立性，因此需要其它机制补偿

## 相关实体

- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]

## 相关概念

- [[KV Cache]]
- [[Double-Wide MLP]]

## 研究备注

- 需要明确区分它和 `Prefix Caching`、`Cache-aware routing` 这类服务层缓存复用机制
