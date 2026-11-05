# 昇腾 A5 相比 A3 升级在哪里，FP16 和 FP8 峰值是多少？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

按 Ascend 950 系列对比 910C，芯片层面的突出变化是新增 FP8 和 FP4 低精度计算能力。华为披露 950 系列 FP8 峰值为 1 PFLOPS。950PR 和 950DT 的 FP16 约 500 TFLOPS 来自第三方规格表，官方逐 SKU 峰值待核实。

## 深入解释

面试时先确认 A5、A3 对应的具体产品及口径，再分开讲芯片精度、HBM 容量与带宽、互联和超节点规模。低精度能否带来收益取决于模型精度方案、实际算子支持及瓶颈。不能拿整机峰值或 FP8 数字回答单芯片 FP16。

## 参考来源与待核实

- [华为发布口径转载](https://feeds-drcn.cloud.huawei.com.cn/landingpage/latest?channel=-1&cpid=666&ctype=news&docid=10510282149091&dy_scenario=relate&ifl=zh_CN&r=CN&tn=09807b2f73903b8458b79766c62b6d377efd03fe6fd540a457a22f8b39d3b6a6&to_app=hwbrowser)。
- [第三方 950 规格表](https://mirrorfrog.com/docs/cards/huawei/ascend-950/)；[讯飞披露材料](https://static.cninfo.com.cn/finalpage/2026-04-29/1225268686.PDF)中 1000 明确标为 FP8。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
