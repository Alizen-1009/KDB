# C++ vector 和 list 如何选择，批量删除怎样更高效？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

vector 连续存储，随机访问 O(1)，遍历通常缓存友好；中间 erase 要移动后缀。list 双向链表，已知迭代器时单点删除 O(1)，查找 O(n)，存在额外节点和指针追踪成本。

## 深入解释

保序批量删除用 C++20 std::erase_if，或 erase(remove_if(begin,end,pred),end)，一次压缩 O(n)。反复从中间 erase 最坏 O(n²)。不要求顺序且已知下标时可移动末元素覆盖，再 pop_back，删除一次 O(1)（按移动操作计）。注意 vector erase 使删除位置及之后的迭代器失效，扩容使全部失效；list erase 只使被删节点的迭代器失效。

## 参考来源与待核实

- [参考 1](https://en.cppreference.com/cpp/container/vector/erase)

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
