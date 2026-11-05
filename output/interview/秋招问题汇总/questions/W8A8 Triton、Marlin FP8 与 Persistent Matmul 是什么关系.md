# W8A8 Triton、Marlin FP8 与 Persistent Matmul 是什么关系？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

W8A8 是输入量化精度，Triton 是实现工具，Marlin 是 kernel 家族，persistent 是 tile 调度方式。具体 dispatcher 可以三选一，算法上 FP8 与 persistent 可以组合。

## 深入解释

vLLM 文档中的 FP8 Marlin 路径为 W8A16，即 FP8 权重配 BF16 或 FP16 激活。该结论限定到文档对应的 Linear 路径。输入、累加、输出精度可以分别选，FP32 累加减少舍入误差，转 BF16/FP16 写回可匹配下游接口并减少输出流量。纯整数 W8A8 的累加口径需要另看具体实现。具体三分支 dispatcher 的源码版本待确认。

## 参考来源与待核实

- [[../../../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]。
- [vLLM FP8 量化文档](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/fp8.md)；[Triton Persistent Matmul](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
