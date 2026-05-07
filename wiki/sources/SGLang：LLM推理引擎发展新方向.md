# SGLang：LLM推理引擎发展新方向

## 来源信息

- 标题：SGLang：LLM推理引擎发展新方向
- 作者：方佳瑞​新知答主
- 日期：2024-07-31
- 类型：文章
- 原始文件：[[../raw/articles/SGLang：LLM推理引擎发展新方向|SGLang：LLM推理引擎发展新方向]]
- 原始链接：[知乎专栏](https://zhuanlan.zhihu.com/p/711378550)

## 2-3 条核心摘要

- 文章把 `SGLang` 的定位从普通 LLM serving engine 扩展到 [[LLM Programs]] 运行框架：它面向多次 LLM 调用、控制流、结构化输入输出、工具使用、多模态输入和受约束生成，而不只是单轮 `prompt -> prefill -> decode -> output`。
- `SGLang` 的前端是嵌入 Python 的 DSL，后端 runtime 重点围绕 [[RadixAttention]]、[[Constrained Decoding]] 和 API speculative execution 优化复杂程序化调用的执行效率。
- 文章认为 `SGLang V2` 的系统调度实现已经在部分 H100 Llama3 serving benchmark 中接近甚至超过 `TensorRT-LLM`，并明显快于 `vLLM`；但作者说明自己没有实测，因此该性能结论应标注为来源声称，待独立复现。

## 值得关注的论断

- `vLLM` 可能处在类似早期 `Caffe` 的阶段：它凭借 `PagedAttention` 和社区生态成为现象级系统，但随着使用范式和硬件环境变化，推理框架仍有继续重构的空间。
- [[LLM Programs]] 的兴起意味着推理系统需要理解更复杂的调用图，而不是只优化单次请求的 `prefill/decode`。
- [[RadixAttention]] 的前缀共享思想已经超出 SGLang Program 内部复用，逐渐成为新一代推理系统中通用的 KV cache prefix sharing 设计参考。
- API speculative execution 展示了 `SGLang` 作为上层调用框架的可能性：即使模型是黑盒 API，也可以通过解释器级别的额外生成和后续匹配减少重复调用。

## 关键概念

- [[LLM Programs]]
- [[RadixAttention]]
- [[Constrained Decoding]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[Speculative Decoding]]
- [[PagedAttention]]
- [[Continuous Batching]]

## 相关实体

- [[../entities/SGLang]]
- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`LLM Programs`、`RadixAttention`、`Constrained Decoding`、`KV Cache`、`Prefix Caching`、`Speculative Decoding`
- 会更新哪些实体页：`SGLang`、`vLLM`、`TensorRT-LLM`
- 是否存在冲突：无直接冲突；但文中对 `SGLang V2` 性能优势的描述来自二手转述和博客图表，应和官方 benchmark、硬件配置、版本参数分开标注。

## 待确认

- `SGLang V2` 相比 `vLLM / TensorRT-LLM` 的性能结论需要补 LMSYS 官方博客、SGLang 论文或本地 benchmark 复核。
- 文中提到 `SGLang` 团队解释性能来自“软件调度写得好”，但没有展开具体调度机制，后续宜补官方设计文档或代码路径。
- API speculative execution 何时启用、失败时代价如何控制，本文没有细讲，暂不应外推为稳定通用收益。
