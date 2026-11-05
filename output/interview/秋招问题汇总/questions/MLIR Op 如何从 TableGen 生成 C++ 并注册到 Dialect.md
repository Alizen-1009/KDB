# MLIR Op 如何从 TableGen 生成 C++ 并注册到 Dialect？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

.td 声明 Op，mlir-tblgen 生成 .h.inc 和 .cpp.inc，C++ 编译器编译 include 这些片段的源码。GET_OP_CLASSES 选择类声明或定义，GET_OP_LIST 选择供 addOperations 使用的 Op 类型列表。

## 深入解释

DialectRegistry 可登记 Dialect 供 MLIRContext 加载。加载 MiniDialect 时，initialize() 通过 addOperations 注册各个 mini.* Op 的定义。构建时生成 C++、运行时加载 Dialect 和为 Op 实现 lowering 是不同阶段。

## 参考来源与待核实

- [MLIR ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)；[MLIR Toy Chapter 2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
