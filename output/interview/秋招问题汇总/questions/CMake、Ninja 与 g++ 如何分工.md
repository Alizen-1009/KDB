# CMake、Ninja 与 g++ 如何分工？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

CMake Configure 解析 CMakeLists.txt 和编译配置，Generate 写出 Ninja 构建规则，Build 由 Ninja 调度 g++ 及其他命令。g++ 编译 C++ 并可驱动链接。

## 深入解释

用 `cmake -S . -B build -G Ninja` 配置和生成，再用 `cmake --build build` 构建。source tree 放手写文件，build tree 放 CMakeCache.txt、build.ninja、生成代码和对象文件。CMake Cache 保存构建目录的持久选项和已发现路径。Target 是可请求构建的目标，Ninja phony 给依赖集合命名，.d 文件记录对象文件依赖的头文件等输入，便于增量构建。

## 参考来源与待核实

- [CMake 命令行](https://cmake.org/cmake/help/latest/manual/cmake.1.html)；[Ninja 手册](https://ninja-build.org/manual.html)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
