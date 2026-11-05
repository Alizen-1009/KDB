# const int 指针的两种写法与常量指针有什么区别？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，C++第 1 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

const int *A 和 int const* B 有什么区别？

## 30 秒回答

两者没有区别，都是指向 const int 的指针，不能通过该指针修改目标值，但指针本身可以改指向。int* const 才是指针本身不可改变；const int* const 则同时限制指针和通过它修改目标。const 约束与对象实际存储位置无关。

## 深入解释

```cpp
int x = 1;
int y = 2;
const int* a = &x;
int const* b = &x;
a = &y;            // 可以：改变指向
// *a = 3;         // 不可以：通过 a 修改目标

int* const p = &x;
*p = 3;            // 可以：修改 x
// p = &y;         // 不可以：改变 p 的指向
```

指向 const 的指针不代表原对象一定声明为 const：x 本身不是 const，仍可通过其他合法途径修改。不要混淆“只读访问路径”和“对象本身不可修改”。

## 参考来源与待核实

- [C++ 草案：cv-qualifiers](https://eel.is/c++draft/dcl.type.cv)。
- 示例是语言规则说明，本批未编译。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/编程、数学与算法|编程、数学与算法]]
- [[../README|秋招问题汇总]]
