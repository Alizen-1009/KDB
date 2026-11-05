# MLIR ODS 合同、类型推导和 Verifier 如何分工？

- 整理状态：整理中
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

ODS 合同声明操作数、属性、结果及其约束。Inference 根据已知输入推导结果类型，Verifier 检查已构造的 Op 是否满足规则。

## 深入解释

以假设的 mini.add 为例，tensor<2x3xf32> 是静态 F32 tensor，tensor<?x3xf32> 有动态维。ranked F32 约束本身允许动态维，静态要求要明确写进约束。两个操作数分别满足静态 F32 仍可能形状不同，同形要求由相应声明式约束、trait 或 verifier 检查。逐个检查四个 mini.* Op 时，应读实际 .td 的 arguments、results 和额外约束。

## 参考来源与待核实

- [MLIR ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)。四个 Op 的实际名称和 .td 定义待提供，示例仅用于解释。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
