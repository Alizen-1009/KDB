# 最多选择 k 台机器如何最大化速度和乘最小可靠性？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，算法题第 6 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

类似 LeetCode 1383：选择最多 maxMachines 台机器，使 sum(speed) * min(reliability) 最大。

## 30 秒回答

可靠性降序扫描，使当前值成为候选最小可靠性；小根堆保留至多 k 个最大 speed，并维护速度和，计算 speed_sum×当前可靠性。最优集合最低可靠性的机器被扫描时，其全部成员已可选，因此能覆盖最优值。假设速度和可靠性非负，乘积要避免溢出。

## 深入解释

标准原题对应 speed/efficiency；这里将 efficiency 换名为 reliability。假设数值为正，maxMachines=k≥1。

```text
按 reliability 降序排序
小根堆 heap，speed_sum=0，answer=0
for (speed, reliability):
    heap.push(speed)
    speed_sum += speed
    if heap.size > k:
        speed_sum -= heap.pop_min()
    answer = max(answer, speed_sum * reliability)
```

扫描到 r 时，已处理机器的可靠性都≥r；堆保留其中至多 k 个最大速度，使速度和最大。考虑最优集合最小可靠性对应机器被处理的时刻，该集合全部可选，堆速度和不更小。

细节：当前机器可能被堆弹出，此时集合实际最小可靠性可能大于 r，当前乘积只是该可行集合目标值的下界；仍不影响通过最优集合边界捕获最优值。

排序 O(n log n)，堆 O(n log k)，堆空间 O(k)，排序/输入存储另计。乘积需按输入上界选足够宽的类型。若题目要求最终 mod 10⁹+7，先比较真实乘积，再对最终答案取模，不能比较取模后的大小。

## 参考来源与待核实

- [LeetCode 1383 官方题目](https://leetcode.com/problems/maximum-performance-of-a-team/description/)。
- 官方题意已在前序对话核对；本页是算法步骤，未运行实现。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/编程、数学与算法|编程、数学与算法]]
- [[../README|秋招问题汇总]]
