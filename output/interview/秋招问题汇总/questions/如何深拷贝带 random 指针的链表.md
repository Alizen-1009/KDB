# 如何深拷贝带 random 指针的链表？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，算法题第 2 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

实现 copyRandomList：带 random 指针链表的深拷贝。

## 30 秒回答

两遍哈希映射最直接：先为每个旧节点创建副本，再用 old→new 映射连接 next/random，时间空间均 O(n)。若要求 O(1) 辅助空间，可把副本穿插到原节点后，借原节点的 next 找到 random 的副本，再拆分并恢复原链表。输出节点不计辅助空间。

## 深入解释

假设 random 指向本链表内部节点或 nullptr，next 构成普通无环链表。

### 哈希表两遍

```text
第一遍：old_to_new[old] = 新建同值节点
第二遍：
    new.next = old.next 为空 ? 空 : old_to_new[old.next]
    new.random = old.random 为空 ? 空 : old_to_new[old.random]
返回 head 的副本；空链表返回空。
```

时间 O(n)，辅助空间 O(n)。必须先建立所有副本，random 可指向后面的节点，不能假设已处理。

### 穿插复制

```text
原链：A -> B -> C
穿插：A -> A' -> B -> B' -> C -> C'
```

对每个 old，copy=old.next，若 old.random 非空，copy.random=old.random.next。随后依次恢复 old.next，并连接 copy.next。

时间 O(n)，辅助空间 O(1)，但会暂时修改原链表，要求可写、无并发观察；若分配失败，还需单独设计恢复与释放。不要把面试骨架宣称为异常安全生产实现。

## 参考来源与待核实

- [LeetCode 138 原题](https://leetcode.com/problems/copy-list-with-random-pointer/)。
- 本页为算法步骤，未运行实现代码。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/编程、数学与算法|编程、数学与算法]]
- [[../README|秋招问题汇总]]
