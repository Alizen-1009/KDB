# LeetCode 2712 如何通过相邻边界推导最小翻转成本？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，算法题第 5 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

LeetCode 2712：动态规划思路。

## 30 秒回答

每次翻转前缀或后缀只改变一条内部相邻边界。最终字符串相同，因此原本不同的边界都必须消除。边界 i 可选成本 i 的前缀翻转或成本 n−i 的后缀翻转，取较小并累加。可写为边界累计式 DP，但关键是证明边界独立，而不是硬套二维状态。

## 深入解释

官方题意：给二进制字符串，可以翻转前缀 s[0..i]，成本 i+1；或翻转后缀 s[i..n−1]，成本 n−i，求最终所有字符相同的最小成本。

将边界 i 定义在 s[i−1] 与 s[i] 之间，1≤i<n。翻转前 i 个字符，只改变边界 i；内部相邻字符同时翻转，相同/不同关系不变。翻转从 i 开始的后缀也只改变边界 i。

所以每条不同边界至少要被改变奇数次，最小成本为 min(i,n−i)；每条相同边界无需改变。独立选择后所有相邻字符相同，满足目标：

$$ans=\sum_{i=1}^{n-1}[s[i-1]\ne s[i]]\min(i,n-i)$$

写成扫描 DP，令 dp[i] 是消除前 i 条相邻差异的最小累计成本：

$$dp[i]=dp[i-1]+[s[i-1]\ne s[i]]\min(i,n-i)$$

```cpp
#include <algorithm>
#include <string>

long long minimumCost(const std::string& s) {
    long long ans = 0;
    int n = static_cast<int>(s.size());
    for (int i = 1; i < n; ++i) {
        if (s[i] != s[i - 1]) ans += std::min(i, n - i);
    }
    return ans;
}
```

时间 O(n)，辅助空间 O(1)。用 long long 存累计值；本公式以最终全 0 或全 1 均可为前提，不能不改动套到固定目标字符串。

## 参考来源与待核实

- [LeetCode 2712 官方题目](https://leetcode.com/problems/minimum-cost-to-make-all-characters-equal/description/)。
- 题意已在本轮对话前序核对；实现为面试示例，本批未编译。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/编程、数学与算法|编程、数学与算法]]
- [[../README|秋招问题汇总]]
