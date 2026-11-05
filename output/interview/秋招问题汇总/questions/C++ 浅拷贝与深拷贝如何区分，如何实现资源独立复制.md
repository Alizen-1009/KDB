# C++ 浅拷贝与深拷贝如何区分，如何实现资源独立复制？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，C++第 4 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

浅拷贝和深拷贝有什么区别？写一个拷贝构造函数，实现深拷贝。

## 30 秒回答

浅拷贝只复制资源地址，两个对象仍指向同一资源；深拷贝为新对象分配独立资源并复制内容。拥有裸指针资源时，要同时设计析构、拷贝构造和拷贝赋值，避免重复释放和悬空指针。实际开发优先使用 vector 等 RAII 成员。

## 深入解释

简化的 int 缓冲区，包含完整的 Rule of Three：

```cpp
#include <algorithm>
#include <cstddef>
#include <utility>

class Buffer {
public:
    explicit Buffer(std::size_t n)
        : n_(n), data_(new int[n]{}) {}

    Buffer(const Buffer& other)
        : n_(other.n_), data_(new int[n_]) {
        std::copy_n(other.data_, n_, data_);
    }

    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            Buffer tmp(other);
            std::swap(n_, tmp.n_);
            std::swap(data_, tmp.data_);
        }
        return *this;
    }

    ~Buffer() { delete[] data_; }

private:
    std::size_t n_;
    int* data_;
};
```

拷贝赋值采用 copy-and-swap：先成功构造副本，再交换；若分配失败，原对象保持不变。示例元素是 int，复制不会抛出元素级异常；不要把它无修改推广为任意 T 的通用容器。

示例未提供移动操作，不是完整高性能容器。使用 `std::vector<int>` 通常即可获得独立拷贝、析构和移动管理，符合 Rule of Zero。

相关：[[C++ 拷贝构造函数是什么，与拷贝赋值有何区别|拷贝构造与赋值]]。

## 参考来源与待核实

- [C++ Core Guidelines：资源与特殊成员函数](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c20-if-you-can-avoid-defining-default-operations-do)。
- 代码用于解释所有权和复制，未在本批编译运行。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/编程、数学与算法|编程、数学与算法]]
- [[../README|秋招问题汇总]]
