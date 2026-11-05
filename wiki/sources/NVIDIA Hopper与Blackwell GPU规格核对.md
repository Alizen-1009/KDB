---
type: source
source_kind: 文章
topic: 性能分析
updated: 2026-11-05
---

# NVIDIA Hopper与Blackwell GPU规格核对

## 来源信息

- 标题：NVIDIA Hopper / Blackwell / Blackwell Ultra 产品规格与数据表核对
- 作者：原厂规格由 NVIDIA 发布；本页为多份一手资料的规格节核对摘要
- 核对日期：2026-11-05；Blackwell Ultra PDF 标记 Oct25
- 类型：产品规格页 / 数据表；本页只收录本次硬件参数主题，不收录产品宣传 benchmark
- 原始文件：外部原始文档 [NVIDIA Blackwell 数据表 PDF（原厂文档的外部镜像，第 8 页）](https://escape-technology.de/images/products/NVIDIA/HGXB300/blackwell-datasheet-4204213.pdf)、[NVIDIA Blackwell Ultra 数据表 PDF（官方托管，第 5 页）](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf)。本轮未向 raw 添加或修改文件；没有伪造本地原始文件路径。
- 补充一手来源：[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)、[NVIDIA H200 规格](https://www.nvidia.com/en-us/data-center/h200/)、[NVIDIA HGX 规格](https://www.nvidia.com/en-us/data-center/hgx/)、[NVIDIA MIG 规格与 profile](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)
- H800 核对线索：[NVIDIA 中文 H800 数据表镜像](https://chaoqing-i.com/upload/20231128/NVIDIA%20H800%20GPU%20Datasheet.pdf)。可检索摘要与 SXM 的 BF16、HBM、互联参数相符，但 PDF 全文获取失败，不能算完成逐项核实。
- 用户输入边界：2026-11-05 对话给出五列完整参数，但缺失型号表头；依数值匹配 H800 SXM、H100 SXM、H200 SXM、HGX B200、HGX B300。对话用于确定整理范围，外部一手资料用于核实；完整表与换算集中维护于 [[../concepts/GPU算力与带宽速查]]。

## 2-3 条核心摘要

- 规格必须绑定具体 SKU、精度和计算路径。H100/H200 的低精度 Tensor Core 规格有稀疏脚注，不能直接拿来估算普通稠密 GEMM。
- HGX B200/B300 单卡均为约 2,250 TFLOPS 稠密 BF16，HBM 带宽均为 7.7 TB/s；显存容量分别为 180 / 270 GB。B300 的 FP64、INT8 与 FP4 稠密规格不同，不能把“代际升级”外推到所有精度。
- $F/W$ 与 Roofline 公式是本次整理的推导；硬件峰值是数据手册规格，两者都不是实测应用性能。

## 值得关注的论断

- 原厂 Blackwell / Ultra 单 GPU TF32 显示约 2.2 PFLOPS；HGX 八卡 TF32 为 18 PFLOPS，换算得到 2.25 PFLOPS/卡，保留取整差异。
- Ultra 单卡 FP4 标为 18 PFLOPS 稀疏 / 14 PFLOPS 稠密；HGX B300 八卡为 144 / 108 PFLOPS，换算单卡稠密为 13.5 PFLOPS。B300 FP4 不能统一按“稠密为稀疏一半”处理。
- Ultra 单卡 INT8 列为 307 TOPS，FP64/FP64 Tensor Core 为 1.2 TFLOPS；保留原厂脚注，不据 FP8 规格类推 INT8 吞吐或指令路径。
- NVLink 和 PCIe 是独立于 HBM 的传输路径，规格中的每卡聚合、双向总量不等于任意 GPU 对或集合通信的有效带宽。

## 关键概念

- [[../concepts/GPU算力与带宽速查]]
- [[../concepts/Roofline 模型]]
- [[../concepts/混合精度训练与推理]]

## 相关实体

- [[../entities/NVIDIA Hopper]]
- [[../entities/NVIDIA Blackwell]]

## 与现有 wiki 的关系

- 现有 [[译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]] 使用 B200 约 7.672 TB/s；本次原厂表按 7.7 TB/s 取整，不视为架构冲突。
- 保留既有架构页内容，新增规格速查入口；在 Roofline 页补充公式、单位与计算路径约束。
- 对前序对话的补充：H800 应明确为 SXM 并标记核实边界；B300 是 Blackwell Ultra；B300 的稠密 FP4 存在特殊比例，不能把“所有低精度稠密减半”作为通则。
- 180 / 270 GB 的 HGX GPU 与 GB200 / GB300 NVL72 中 186 / 279 GB、8 TB/s 的 GPU 是不同配置，不把它们覆盖成单一“最新值”。

## 待确认

- H800 SXM 原厂 PDF 的完整读取及 FP64 等各行独立核实；目前保留用户原表值与可检索摘要的证据边界。
- INT8 实际可用指令、Tensor Core / 其他执行路径与吞吐需绑定 GPU target、CUDA/PTX、kernel；本页仅记录数据手册标称行。
- 所有设备的持续带宽、计算利用率、P2P / collective 有效带宽与 LLM 端到端收益均未运行 benchmark。
