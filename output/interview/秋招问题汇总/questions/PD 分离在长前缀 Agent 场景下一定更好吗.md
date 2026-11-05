# PD 分离在长前缀 Agent 场景下一定更好吗？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 6 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

PD 分离一定好吗？比如在 Agent 场景下 prefix cache 很长？

## 30 秒回答

不一定。PD 分离通过状态交接换取阶段隔离和独立调优；Agent 若长前缀已在 Decode worker 命中、新增输入很短，远端 Prefill 的状态获取与交接可能比本地续算更贵。应结合缓存位置、未命中输入长度、排队和干扰选择路径，而不是让所有请求强制远端 Prefill。

## 深入解释

例：Decode 侧已保存 100K tokens 会话历史，工具返回只新增 100 tokens。远端 Prefill worker 若没有历史状态，可能需要远程读取、迁移或重算，然后再次交接；这个例子只表达形态，不是实测数据。

应比较：

$$T_{local}=T_{queue,local}+T_{incremental,local}$$
$$T_{remote}=T_{queue,P}+T_{state\ acquisition}+T_{prefill,P}+T_{handoff}$$

同时考虑本地 Prefill 对其他 Decode 请求 TPOT 的干扰。长前缀本身不意味着分离必差：若缓存已可共享、两侧都命中、只迁移增量或网络很快，代价会不同。

工程方案包括缓存亲和路由、按未命中长度决定本地/远端、共享缓存层、增量传输和请求准入。长且未命中的 Prefill 更可能受益于独立资源池；高命中短增量可优先评估本地续算。

## 关联知识

- [[../../../../wiki/concepts/PD分离|PD分离]]
- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]
- [[../../../../wiki/concepts/缓存感知路由|缓存感知路由]]

## 参考来源与待核实

- 机制依据见关联 wiki；本文成本模型为工程归纳。不能从单个例子推出固定分离阈值；最优策略需按流量、互联和 backend 测量。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
