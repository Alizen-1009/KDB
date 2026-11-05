---
type: concept
topic: 性能分析
updated: 2026-11-05
sources: 1
---

# GPU算力与带宽速查

## 范围与阅读口径

汇总 H800、H100、H200、B200、B300 的峰值算力、显存容量、HBM 带宽、MIG 与互联，供 [[Roofline 模型]] 和 LLM 性能估算使用。核对日期：2026-11-05；出处和未核实项见 [[../sources/NVIDIA Hopper与Blackwell GPU规格核对]]。

> [!important] 先绑定 SKU 和计算路径
> 下表均为**单 GPU**。Hopper 三列按 SXM，Blackwell 两列按 HGX 中的单 GPU；不能与 H100/H200 NVL、PCIe 版或 GB200/GB300 NVL72 中的 GPU 混用。H800 列是依据用户原表参数匹配出的 SXM 口径，尚未完整读取其原厂 PDF，逐项独立核实仍待完成。
>
> $F$ 必须带精度、CUDA Core / Tensor Core、稠密 / 稀疏限定；$W$ 在本文指 **HBM 显存带宽**。它不是 NVLink、PCIe 或网卡带宽。峰值规格不是实测可持续性能。

## 完整硬件参数

保留用户提供的全部指标，补齐型号与脚注。除特殊标注外，低精度 Tensor Core 行是**结构化稀疏峰值**，不是普通稠密 GEMM 的峰值。FP64、FP64 Tensor Core 和普通 FP32 行不套用稀疏减半规则。

| 指标 | H800 SXM¹ | H100 SXM | H200 SXM | HGX B200 单 GPU | HGX B300 单 GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 架构 | Hopper | Hopper | Hopper | Blackwell | Blackwell Ultra |
| FP64（TFLOPS） | 1 | 34 | 34 | 37 | 1.2 |
| FP64 Tensor Core（TFLOPS） | 1 | 67 | 67 | 37 | 1.2 |
| FP32（TFLOPS） | 67 | 67 | 67 | 75 | 75 |
| TF32 Tensor Core（TFLOPS，稀疏） | 989 | 989 | 989 | 2,250² | 2,250² |
| FP16 Tensor Core（TFLOPS，稀疏） | 1,979 | 1,979 | 1,979 | 4,500 | 4,500 |
| BF16 Tensor Core（TFLOPS，稀疏） | 1,979 | 1,979 | 1,979 | 4,500 | 4,500 |
| FP8 Tensor Core（TFLOPS，稀疏） | 3,958 | 3,958 | 3,958 | 9,000 | 9,000 |
| INT8 Tensor Core（TOPS，稀疏规格脚注） | 3,958 | 3,958 | 3,958 | 9,000 | 307³ |
| FP4 Tensor Core（TFLOPS，稀疏） | — | — | — | 18,000 | 18,000⁴ |
| GPU 显存容量（GB） | 80 | 80 | 141 | 180 | 270 |
| GPU 显存带宽（TB/s） | 3.35 | 3.35 | 4.8 | 7.7 | 7.7 |
| MIG 最大实例数 | 7 | 7 | 7 | 7 | 7 |
| NVLink 总带宽（GB/s，双向合计） | 400 | 900 | 900 | 1,800 | 1,800 |
| PCIe 代际 | Gen5 | Gen5 | Gen5 | Gen5 | Gen6 |
| PCIe x16 带宽（GB/s，双向合计，规格取整） | 128 | 128 | 128 | 128 | 256 |

出处：H100、H200 分别核对 NVIDIA 产品规格页；B200 核对 Blackwell 数据表第 8 页，B300 核对 Blackwell Ultra 数据表第 5 页及 HGX 规格页，链接集中在来源页。

1. H800 数值来自用户表；可检索到的 NVIDIA 中文数据表镜像摘要与 SXM 的 BF16、带宽、互联相符，但本轮 PDF 全文获取失败，保留待核实标记。不要拿 H800 PCIe 的 2 TB/s、较低计算峰值与该列拼接。
2. Blackwell 单 GPU PDF 将 TF32 显示为约 2.2 PFLOPS；HGX 整机稀疏 TF32 为 18 PFLOPS，除以 8 得 2.25 PFLOPS，即原表的 2,250 TFLOPS。差异属于规格展示的舍入精度。
3. B300 的 307 TOPS 是 Blackwell Ultra 数据表列出的规格，明显低于 B200 的 9,000 TOPS；不能根据两者 FP8 相同而补写相同 INT8 峰值。该行沿用数据表的稀疏脚注；具体指令路径和实测吞吐仍需按工具链验证。INT8 使用 TOPS，不写 TFLOPS。
4. B300 FP4 的稠密峰值不能简单取稀疏的一半，见下节。

## 稠密峰值：用于普通矩阵计算的口径

以下是对上述规格脚注的换算。结构化稀疏要求数据满足硬件支持的模式并由对应 kernel 使用；MoE 只激活部分专家、任意零元素、稀疏 Attention 都不自动满足这一条件。乘加 FMA 通常计为 2 FLOP；输入精度与累加精度应分别核对。

| 稠密计算路径 | H800 SXM¹ | H100 SXM | H200 SXM | HGX B200 单 GPU | HGX B300 单 GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF32 Tensor Core（TFLOPS） | ≈494.5 | ≈494.5 | ≈494.5 | ≈1,125 | ≈1,125 |
| FP16 Tensor Core（TFLOPS） | ≈989.5 | ≈989.5 | ≈989.5 | 2,250 | 2,250 |
| BF16 Tensor Core（TFLOPS） | ≈989.5 | ≈989.5 | ≈989.5 | 2,250 | 2,250 |
| FP8 Tensor Core（TFLOPS） | 1,979 | 1,979 | 1,979 | 4,500 | 4,500 |
| INT8（TOPS，按该行脚注换算） | 1,979 | 1,979 | 1,979 | 4,500 | ≈153.5 |
| FP4 Tensor Core（TFLOPS） | — | — | — | 9,000 | ≈13,500² |

1. H800 继承原表待核实状态。Hopper BF16 常简记为 **约 989–990 TFLOPS**；由已舍入的 1,979 减半得到 989.5，不能视为更精确的原厂时钟计算结果。
2. NVIDIA HGX 页给出 B300 八卡 FP4 **144 PFLOPS 稀疏 / 108 PFLOPS 稠密**，折合单卡 **18 / 13.5 PFLOPS**；Ultra 单卡 PDF 显示 **18 / 14 PFLOPS**，后者是较粗的舍入展示。B200 八卡对应 144 / 72 PFLOPS，单卡 18 / 9 PFLOPS。因此 B300 的稠密 FP4 约提升 50%，即使两代的稀疏 FP4 数字相同。

普通 FP32 峰值只适用于相应 CUDA Core 计算路径；不能用 BF16 Tensor Core 峰值估算 FP32 reduction、地址运算或任意标量 kernel。TF32 Tensor Core 的吞吐也不代表严格 IEEE FP32 运算具有同等吞吐。低精度格式及 scale 开销见 [[混合精度训练与推理]]。

## 用 F/W 判断瓶颈

设运算量为 $C$ FLOP，实际穿过 HBM 接口的数据量为 $D$ Byte，算术强度为 $I=C/D$。理想 Roofline 上界与时间下界为：

$$
P_{\max}(I)=\min(F,WI),\qquad
T\gtrsim\max\left(\frac{C}{F},\frac{D}{W}\right),\qquad
I_{\mathrm{ridge}}=\frac{F}{W}.
$$

- $I<F/W$：HBM 带宽上界更低，优先考虑减少流量、融合和提高复用。
- $I>F/W$：计算上界更低，优先检查对应计算管线的利用率和 shape；不代表实际已经达到 compute-bound。
- 两者相近：计算与访存都应关注；模型不包含全部 launch、同步、延迟、通信与尾部开销。

统一采用**稠密 FP16/BF16 Tensor Core** 的 $F$，按十进制 $1\,\mathrm{TFLOPS}=10^{12}\,\mathrm{FLOP/s}$、$1\,\mathrm{TB/s}=10^{12}\,\mathrm{Byte/s}$ 换算：

| GPU | 显存（GB） | $W$（TB/s） | $F$（TFLOPS，约） | $F/W$（FLOP/Byte，约） |
| --- | ---: | ---: | ---: | ---: |
| H800 SXM（待核实） | 80 | 3.35 | 989.5 | 295 |
| H100 SXM | 80 | 3.35 | 989.5 | 295 |
| H200 SXM | 141 | 4.8 | 989.5 | 206 |
| HGX B200 单 GPU | 180 | 7.7 | 2,250 | 292 |
| HGX B300 单 GPU | 270 | 7.7 | 2,250 | 292 |

例如 H100 上一个 BF16 Tensor Core workload 的 HBM 算术强度为 100 FLOP/Byte，则理想吞吐上界为 $\min(989.5,3.35\times100)=335$ TFLOPS。这是模型计算，不是 benchmark。若改走 FP32 CUDA Core，应改用约 67 TFLOPS，对应分界点约 20 FLOP/Byte。

$D$ 必须与所选带宽层级一致：L2 命中会改变 HBM 流量，不能直接把逻辑 tensor 字节数当作实际 HBM 字节数，也不能拿 HBM 的 $D$ 除以 NVLink 带宽。详细机制见 [[Roofline 模型]]、[[Profiling]]。

## 容量、显存带宽与互联分别影响什么

| 指标 | 主要约束 | 估算边界 |
| --- | --- | --- |
| 显存容量 | 权重、KV Cache、激活、workspace 能否放下，可容纳的 batch / context | GB 与 GiB、框架保留空间和实际可分配量需分开 |
| HBM 带宽 | 本地显存数据供给，权重和 KV 读取等 | 数据手册峰值不等于有效带宽；cache、访问模式和利用率会改变结果 |
| NVLink | 机内 GPU 间数据交换与集合通信 | 表中是每 GPU 全部链路的双向合计；不是任意一对 GPU 的保证带宽，也不是 all-reduce 有效带宽 |
| PCIe | GPU 与主机或其他 PCIe 设备间传输 | x16 Gen5 约 64 GB/s 单向、128 GB/s 双向；Gen6 规格约 128 / 256，实际 payload 另有协议与拓扑开销 |
| MIG | 将一个 GPU 分成隔离实例 | “最多 7 个”不意味着同时拥有 7 份整卡算力或 7 倍显存；profile、可用显存与资源分配有约束 |

NVLink 双向合计折半可得到理想发送方向的**聚合规格**：H800 约 200 GB/s、H100/H200 约 450 GB/s、B200/B300 约 900 GB/s；实际点对点与集体通信仍要看链路拓扑和实现。HBM 带宽使用其自身公布口径，不按 NVLink 的规则再减半。

## 代际比较与速记

以下是规格比值推导，不是应用加速比：

- **H800 与 H100**：按用户表，两者 BF16/FP8 峰值和本地显存容量、带宽相同；H800 的 NVLink 从 900 降至 400 GB/s，FP64 从 34 / 67 降至约 1 / 1 TFLOPS。H800 逐项原厂核实尚待完成。
- **H100 → H200**：上述计算峰值基本相同；容量为 $141/80\approx1.76$ 倍，带宽为 $4.8/3.35\approx1.43$ 倍，BF16 ridge point 从约 295 降至 206 FLOP/Byte。
- **H200 → B200**：稠密 BF16 峰值约 $2250/989.5\approx2.27$ 倍，带宽约 $7.7/4.8\approx1.60$ 倍，容量约 1.28 倍，NVLink 聚合带宽约 2 倍。
- **B200 → B300**：容量从 180 到 270 GB，增加 50%；表内 HBM、NVLink 和 BF16/FP8 峰值基本相同，稠密 FP4 约增加 50%，PCIe 从 Gen5 到 Gen6；FP64 和 INT8 规格反而明显降低，不能概括为所有精度都更快。
- 小 batch Decode 的权重读取、长上下文 KV 读取常更受容量/带宽影响；大 GEMM 更有机会受计算峰值限制，但应按 batch、shape、dtype、缓存复用和 kernel 路径判断，不能只按 Prefill / Decode 名称判定。

## 相关页面

- [[Roofline 模型]]
- [[混合精度训练与推理]]
- [[CUDA内存层次]]
- [[../entities/NVIDIA Hopper]]
- [[../entities/NVIDIA Blackwell]]
- [[../sources/NVIDIA Hopper与Blackwell GPU规格核对]]
