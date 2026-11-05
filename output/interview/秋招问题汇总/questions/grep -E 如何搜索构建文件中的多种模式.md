# grep -E 如何搜索构建文件中的多种模式？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

-E 启用扩展正则，竖线表示或，圆括号分组，加号表示重复一次或多次。用单引号保护模式，让 shell 原样传给 grep。

## 深入解释

例如 `grep -nE 'mlir-tblgen|MiniOps(\.h|\.cpp)?\.inc' build/build.ninja` 会打印命中行及行号。反斜杠转义点，问号表示前一组可出现零次或一次。

## 参考来源与待核实

- [GNU grep 正则说明](https://www.gnu.org/software/grep/manual/html_node/Regular-Expressions.html)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
