# 英伟达 AI Infra 五面面经

- 题目出处：用户提供的三张面经截图；题目出现信息来自截图，答案依据单独列在问题页。
- 收录日期：2026-11-05；实际面试日期、职位细分及各题对应轮次未提供，不从“五面”标题推断轮次分配。

按截图原顺序整理全部 **37 题**：AI Infra 与推理系统 12 题、硬件与算子 9 题、C++ 6 题、数学 3 题、算法 7 题。复用已有同义题，答案只在 questions/ 问题页维护；每页保留原题、短答、必要推导或代码及适用边界。整理状态不代表事实全部核实或用户已掌握。

注意原题边界：算法第 1 题缺少替换规则；算法第 4 题截图右侧裁切且“空间”未定义；MLA 的“激活值很大”及数字字符串“求总和”均保留解释假设。相关题页明确标注，不能把条件性示例当成确定题意。代码与性能限制以各题页说明为准。

## AI Infra 与推理系统

1. [[../questions/如何用 Roofline 推导 GEMM 与 MoE 的临界 token 数|根据硬件计算和访存带宽，计算 Matmul 算子的最佳输入数据大小，再根据这个计算 MoE TopK 输入的最佳 token 数（Roofline 模型）。]]
2. [[../questions/FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么|讲一下 FlashAttention？FA 怎么实现的？FlashDecoding 是什么？]]
3. [[../questions/Prefix Caching 如何复用前缀，命中后 attention 执行有何变化|PrefixCache 的作用是什么？和 KVCache 有什么区别？怎么实现？]]
4. [[../questions/MLA Chunked Prefill 如何避免历史 KV 展开导致激活显存过大|Chunk Prefill 在 MLA 中，KV load 之后做 proj，如果 KV 很大导致激活值很大怎么办？]]
5. [[../questions/Agent 场景 KV Cache 空间不足时如何优化|针对 Agent 场景，KVCache 空间不够，有没有什么优化？]]
6. [[../questions/PD 分离在长前缀 Agent 场景下一定更好吗|PD 分离一定好吗？比如在 Agent 场景下 prefix cache 很长？]]
7. [[../questions/Context Parallel 中 AllGather KV 与 AllGather Q 如何选择|CP AG KV 和 AG Q 方案怎么选择？Agent 场景还能 AG KV 吗？]]
8. [[../questions/Decode 长输出请求如何做负载均衡与优化|Decode 如果输出长，怎么做负载均衡？怎么优化？]]
9. [[../questions/为什么 MoE 部署要开大 EP，吞吐收益取决于什么|为什么要大 EP、大 DP？大 EP 开多大合适？固定单卡的 batch size 为 16，以 DeepSeek 为例，计算 EP 大小？]]
10. [[../questions/PD 分离中的 Prefill 与 Decode 实例配比如何确定|PD 分离配比怎么定？Decode 多大 batch size 合适？怎么由此反推 Prefill 的数量？]]
11. [[../questions/MoE 的专家并行 EP 如何工作，核心通信与系统难点是什么|MoE 并行的 overhead 是什么？A2A 通信量是多少？]]
12. [[../questions/四台服务器如何在内存受限时并行计算矩阵乘法|4 台服务器，每台 memory 大小是 N*N，服务器之间网络全连接，计算 C=A*B，A 和 B 都是 N×N 矩阵，怎么并行？如果缓存空间只有 N*N/8，矩阵乘并行方案怎么做？]]

## AI Infra 硬件与算子

1. [[../questions/MQA 和 GQA 为什么比 MHA 更节省 KV Cache|Attention 变化：MHA、MQA、GQA、MLA，对算子的影响是什么？计算访存比如何变化？]]
2. [[../questions/GPU 算子有哪些优化方法，如何按瓶颈选择|讲一下算子有哪些优化方法？]]
3. [[../questions/如何根据 GPU 硬件特性制定 Infra 优化方案|在 Infra 过程中怎么根据硬件特性进行优化？]]
4. [[../questions/相同输入的算子为何可能不确定，atomicAdd 会带来什么问题|同样的输入，算子会出现随机的输出吗？AtomicAdd 等操作顺序随机会带来什么问题？]]
5. [[../questions/FP32 大数减极小数为何不变，如何区分舍入与溢出|FP32 下 9.9*10^9 - 1.0*10^-9 等于多少？会溢出吗？结果是什么？]]
6. [[../questions/GPU 有哪些内存与缓存，地址空间和物理层次如何区分|讲一下 GPU 有哪些 memory？]]
7. [[../questions/两个进程如何共享 GPU 显存，GPU 是否有虚拟地址|两个进程能同时操作同一个 GM 地址吗？GM 有虚拟地址吗？]]
8. [[../questions/CUDA 算子的常量参数放在哪种内存中|算子中的常量参数是放在哪个 memory 的？]]
9. [[../questions/单核上两个线程何时可能比一个线程更快|单核什么情况下会出现两个线程比一个线程的耗时更短？]]

## C++

1. [[../questions/const int 指针的两种写法与常量指针有什么区别|const int *A 和 int const* B 有什么区别？]]
2. [[../questions/两个指针指向常量时如何交换它们的指向|A 指向 1，B 指向 2，交换 A 和 B，1 和 2 是常量。]]
3. [[../questions/C++ 拷贝构造函数是什么，与拷贝赋值有何区别|C++ 拷贝构造函数是什么？]]
4. [[../questions/C++ 浅拷贝与深拷贝如何区分，如何实现资源独立复制|浅拷贝和深拷贝有什么区别？写一个拷贝构造函数，实现深拷贝。]]
5. [[../questions/虚函数与继承如何影响 C++ 对象的 sizeof|考察虚函数和继承：对象大小、虚函数表、继承关系分别如何影响 sizeof？]]
6. [[../questions/不用临时变量交换两个 int 时加减法与异或法有什么边界|不用额外空间，交换两个 int 变量的值：加减法和异或法分别怎么做？]]

## 数学题

1. [[../questions/如何求点到直线或线段的投影点|点 A 和 B 形成向量，点 C 到向量 AB 的投影点 C′，求 C′ 的位置。]]
2. [[../questions/Sigmoid 函数如何求导并稳定计算|Sigmoid 函数是什么？怎么求导？]]
3. [[../questions/Softmax 为什么减去最大值，数值稳定性边界是什么|Softmax 函数是什么？为什么要减 max？]]

## 算法题

1. [[../questions/排序二叉树中替换分数相同节点的题意如何澄清|排序二叉树，替换分数相同的节点。]]
2. [[../questions/如何深拷贝带 random 指针的链表|实现 copyRandomList：带 random 指针链表的深拷贝。]]
3. [[../questions/两个空瓶换一瓶汽水时 N 元最多喝多少瓶|一块钱买一瓶汽水，两个空瓶子换一瓶汽水，给 N 块钱，能喝多少瓶？]]
4. [[../questions/耗时为一到 N 的独立任务如何最小化完工时间与核数|1～N 个任务，任务 i 耗时 i，给出需要多少个核并行去做，才能让执行时间、空间最短？]]
5. [[../questions/LeetCode 2712 如何通过相邻边界推导最小翻转成本|LeetCode 2712：动态规划思路。]]
6. [[../questions/最多选择 k 台机器如何最大化速度和乘最小可靠性|类似 LeetCode 1383：选择最多 maxMachines 台机器，使 sum(speed) * min(reliability) 最大。]]
7. [[../questions/数字字符串插入任意加号后的所有表达式之和如何计算|数字字符串，插入任意加号，求总和。]]

## 返回

- [[../README|秋招问题汇总]]
