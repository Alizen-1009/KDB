---
title: "逆向 NVIDIA：nv_isa_solver 用模糊测试把 nvdisasm \"榨干\"成 NVIDIA GPU ISA 手册"
source: "https://mp.weixin.qq.com/s/fSbMaETut3te2pgozw9rng"
author:
  - "[[nv_isa_solver]]"
published:
created: 2026-06-09
description: "通过对 nvdisasm 位级模糊测试，逐比特反推 NVIDIA GPU 指令编码、操作数与修饰符语义，结合活跃区间分析自动生成 SM89 与 SM90a 可读 ISA 规范文档。"
tags:
  - "clippings"
---
nv\_isa\_solver *2026年5月31日 13:00*

关键词： **NVIDIA GPU** 、 ***指令集逆向*** 、差分模糊测试、 **nvdisasm** 、 ***SASS 编码***

,15分钟

> 写在前面：NVIDIA 从未公开过 SASS（Streaming ASSembler）指令集的完整编码格式。每一代架构升级——从 Volta、Turing、Ampere 到 Hopper——CUDA Binary Utilities 文档里只附一份"助记符列表"，连每条指令多少位、操作数怎么编码、修饰符在哪几位都讳莫如深。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH8AzPA0oPsTBFtSLTZGOaX3lH9DK6BAPRKgjuRxOCaaG6ogTBAPuZltguFRcgicWZuNmZZD2clX32sjIMVrG7NNQKfsicV53mQCs/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

- **Nvidia Instruction Set Specification Generator**
- 项目代码：https://github.com/kuterd/nv\_isa\_solver/tree/main
- SM90a 在线版：https://kuterdinel.com/nv\_isa/
- SM89 在线版：https://kuterdinel.com/nv\_isa\_sm89/

,15分钟

相关推荐

- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)
- [逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)
- [取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)

> 但 CUDA 工具链自带的 `nvdisasm` 却"知道一切"：你喂给它任意 16 字节，它就能告诉你这是什么指令。 `nv_isa_solver` 抓住的，正是这条 **信息不对称** 的缝隙——既然反汇编器是一个黑盒 oracle，那就 **把它当作可查询的真值表** ，用差分模糊测试把每一比特的语义"问"出来。

本文将沿着代码主干，拆解它如何在不到 1500 行 Python 中，将一台 H100 的指令集还原成一份带颜色、可点击的 HTML 手册（SM90a 在线版 <sup>[1]</sup> 、SM89 在线版 <sup>[2]</sup> ）。

![这是基于本项目产出的一份英伟达Hopper架构（SM90a）指令集的逆向工程文档，由模糊测试反汇编工具自动生成，将公开求解器源码。文档列出了NOP、LEA、VABSDIFF等多条GPU指令，为研究其底层指令集提供了一手资料。详见：](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicQIIEPvwic43oYribgDXUaVCdkLHicb1q7jKDM44qcEYUpu26UYia7GqAxvrXhInVh2LDUUFrgBQicYDzuxSbL5SWWVvBAaZCMdqxo/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

这是基于本项目产出的一份英伟达Hopper架构（SM90a）指令集的逆向工程文档，由模糊测试反汇编工具自动生成，将公开求解器源码。文档列出了NOP、LEA、VABSDIFF等多条GPU指令，为研究其底层指令集提供了一手资料。详见：

![这是一条GPU汇编指令的编码说明：指令格式为，含1个谓词P、5个寄存器R、2个立即数和1个数组操作数，对应不同读写端口与编码字段，还包含修饰符组定义，用于纹理采样操作。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicTDLJf3JQfic4rCR4R6YohbYPibOHr7c1m5E3VZWTsOoRI0riaEgf75C9UaWts62x8CyQrPiatQHRwOes4aAEqZv3ibc0lECrsiasYo/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

这是一条GPU汇编指令的编码说明：指令格式为，含1个谓词P、5个寄存器R、2个立即数和1个数组操作数，对应不同读写端口与编码字段，还包含修饰符组定义，用于纹理采样操作。

## 本文目录

- 快速上手
- 一、整体思路：把反汇编器当成"可微的黑盒"
- 二、采集与蒸馏：先攒一个高质量的指令"种子库"
- 2.1 用 nvdisasm 当 oracle 的最小封装
	- 2.2 指令蒸馏：把指令"剃光"到只剩骨架
- 三、位级模糊：从 128 次翻转到一张编码表
- 3.1 翻转每一位，观察发生了什么
	- 3.2 \`InstructionMutationSet\`：把翻转结果归类成"比特集合"
	- 3.3 多个分析 pass：修复一阶模糊的"误判"
- 四、字段重建：从比特集合到 \`EncodingRanges\`
- 4.1 枚举每个字段的具体含义
- 五、活跃区间分析：让每个操作数知道自己是"读"还是"写"
- 六、把一切产出渲染成可读的 HTML 手册
- 七、为什么这件事重要：差分模糊测试的一次教科书式应用
![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GxIgp4icchHibQc5Mlr5FgQHGEHL9iavXNZRUWemDSiblLQJ3o7ZN8ERSKXbATiaqByvcicNB1C4sNu95ia0ZkzFczAsdE7Sic8nHxxelsQwW1FrHFE/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 快速上手

> 项目本身是一个标准 Python 包，依赖只有 `tqdm` ，但 **前置条件** 是本机能直接调用 CUDA Toolkit 中的 `nvdisasm` （路径可通过 `--nvdisasm` 指定）。

仓库里已经附带了一份预先采集好的 `disasm_cache.txt` （约 28 MB，包含上万条已反汇编的指令样本），所以即使你没有 H100，也能跑通整条分析链路。

```
# 1. 安装
git clone https://github.com/kuterd/nv_isa_solver
cd nv_isa_solver
pip install -e .

# 2. 直接用仓库自带的 cache，一键生成 SM90a 的 HTML 手册
nv-isa-solver --arch SM90a --arch_code 90 \
              --cache_file disasm_cache.txt \
              --num_parallel 8
#   产物：output/index.html + output/<OPCODE>.html，外加 isa.json

# 3. 想从零开始采样？先用枚举 + 已知 SASS 反汇编扩展 cache
nv-isa-solver-populate-cache --arch SM90a   # 枚举低 12 位 opcode
cuobjdump --dump-sass --gpu-architecture sm_90 your_kernel.cubin > sass.txt
nv-isa-solver-scan --arch SM90a sass.txt    # 把真实 cubin 中的指令注入 corpus
nv-isa-solver-mutate --arch SM90a           # 已知 opcode × 已知 seed 交叉变异
```

如果只想看最终成品，直接在浏览器中打开 `output/index.html` 即可，每条指令都会渲染成上文图示那样的 16 字节比特表。关于活跃区间分析需要的 cubin 生成器细节，可参考 TuringAs <sup>[4]</sup> ；指令解析层借鉴自 CuAssembler <sup>[5]</sup> 。

## 一、整体思路：把反汇编器当成"可微的黑盒"

> 整套系统可以用一句话概括 **：枚举所有可能影响一条指令语义的比特，逐个翻转，观察反汇编结果的差异，然后用规则把差异归纳成"编码字段"。**

四个核心阶段的数据流如下：

| 阶段 | 入口脚本 | 关键能力 |
| --- | --- | --- |
| ① 语料采集 | `populate_cache.py`  / `scan_disasm.py` / `mutate_opcodes.py` | 枚举 opcode、扫描真实 cubin、交叉变异，喂给 `nvdisasm` 攒一份 `disasm_cache.txt` |
| ② 指令蒸馏 | `Disassembler.distill_instruction` | 把一条采到的指令中"无关紧要"的比特全置 0，得到该签名的 **最小骨架** |
| ③ 位级模糊 | `InstructionMutationSet`  \+ 一组 `analysis_*` pass | 翻转 128 比特中的每一位，对照基准 disasm，分类出 opcode/操作数/修饰符/flag/控制码 |
| ④ 字段归并 + 寿命分析 | `EncodingRanges`  \+ `life_range.py` | 把零散比特聚成连续字段，再借助 nvdisasm 的 `--print-life-ranges` 推断每个操作数是 R/W/RW |

最终所有结果序列化到 `isa.json` （约 5.5 MB），并渲染为 HTML 表格。整个流程的精妙之处在于： **它完全不依赖任何 NVIDIA 的内部文档，唯一的"知识源"就是 `nvdisasm` 这个二进制 oracle** 。

## 二、采集与蒸馏：先攒一个高质量的指令"种子库"

### 2.1 用 nvdisasm 当 oracle 的最小封装

> `Disassembler` 类是整个系统的水龙头。它的核心是把"调用一次 `nvdisasm` 解析 16 字节指令"做成可缓存、可并行批处理的接口。

注意其中两个关键设计： **结果缓存** （避免重复 fork 子进程）和 **并行批处理** （用 `subprocess.Popen` 同时拉起 CPU 核心数那么多个 `nvdisasm` ）。

```
# 来源：nv_isa_solver/disasm_utils.py
def disassemble(self, inst):
    inst = bytes(inst)
    if inst in self.cache:                 # 命中缓存，零成本
        return self.cache[inst]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(inst); tmp.close()
    result = subprocess.run(
        [self.nvdisasm, tmp.name, "--binary", self.arch],
        capture_output=True)
    os.remove(tmp.name)
    result = _process_dump(result.stdout.decode("ascii"))
    self.cache[inst] = result
    return result
```

为什么必须缓存？因为后面每分析一条指令，都要做 **上百次乃至上千次比特翻转 + 重新反汇编** 。仓库里那份 29 MB 的 `disasm_cache.txt` ，就是从冷启动一路喂出来的"知识库"。

### 2.2 指令蒸馏：把指令"剃光"到只剩骨架

> 直接从真实 cubin 抓到的 SASS 指令带有寄存器编号、立即数、控制码、reuse mask 等大量噪音，如果直接拿来翻转每一位，分析逻辑会被淹没在杂讯里。 `distill_instruction` 用一个 **贪心的位翻转算法** 解决这个问题：

```
# 来源：nv_isa_solver/disasm_utils.py
def distill_instruction(self, inst):
    original_asm = self.disassemble(inst)
    original_parsed = InstructionParser.parseInstruction(original_asm)

    distilled = bytes(inst)
    for i in range(127, -1, -1):                 # 从高位往低位尝试
        inst_ = bytearray(bytes(distilled))
        if (inst_[i // 8] >> (i % 8)) & 1 == 0:
            continue                              # 已经是 0 就跳过
        inst_[i // 8] &= ~(1 << (i % 8))          # 试探性清零
        distill_asm = self.disassemble(inst_)
        ifnot distill_asm: continue
        distill_parsed = InstructionParser.parseInstruction(distill_asm)
        if original_parsed.get_key() != distill_parsed.get_key():
            continue                              # 改变了指令"签名"则回退
        distilled = bytes(inst_)                  # 否则采纳清零
    return distilled
```

"签名"由 `Instruction.get_key()` 提供——它是指令基名 + 操作数类型序列的拼接。换言之， **只要反汇编出来还是"同一类"指令，就把这一位清零** 。蒸馏完成后，得到的是一条 **几乎全 0、但仍能被 `nvdisasm` 识别为该指令** 的最小编码骨架，非常适合后续做差分分析。

## 三、位级模糊：从 128 次翻转到一张编码表

### 3.1 翻转每一位，观察发生了什么

得到骨架后， `mutate_inst` 会对 `[start, end)` 范围内的每一位都生成一个"翻转一位"的副本，并批量送进 `nvdisasm` ：

```
# 来源：nv_isa_solver/disasm_utils.py
def mutate_inst(self, inst, start=0, end=16 * 8):
    idxes, insts = [], []
    for i in range(start, end):
        inst_ = bytearray(bytes(inst))
        inst_[i // 8] ^= (1 << (i % 8))     # 仅翻转第 i 位
        insts.append(inst_); idxes.append(i)
    return zip(idxes, insts, self.disassemble_parallel(insts))
```

主流水线里调用的是 `mutate_inst(inst, end=14 * 8 - 2)` ——即只翻转前 13 字节再多一点点，把最后 ~2.5 字节留给控制码（stall、yield、barrier、reuse mask）单独处理。这是因为控制码在所有指令里位置固定，规则化即可，没必要再去模糊。

### 3.2 InstructionMutationSet：把翻转结果归类成"比特集合"

这是整个工程最核心的一段逻辑：它维护了若干个 **比特集合** ，按照"翻这一位会让 disasm 怎样改变"把比特分门别类：

```
# 来源：nv_isa_solver/instruction_solver.py
class InstructionMutationSet:
    def __init__(self, inst, disasm, mutations, disassembler):
        ...
        self.opcode_bits = set()              # 翻转后指令签名变了 → 这是 opcode 区
        self.operand_value_bits = set()       # 操作数的"值"变了
        self.operand_modifier_bits = set()    # 操作数附带的 .modifier 变了
        self.modifier_bits = set()            # 指令级 modifier 变了
        self.predicate_bits = set()           # 谓词寄存器变了
        self.bit_to_operand = {}              # 这一位影响的是第几个操作数
        # …还有一些 flag 候选位
        self._analyse()
```

`_analyse()` 的判定逻辑非常优雅：对每一个 `(i_bit, mutated_inst, mutated_asm)` 三元组， **重新解析 asm，并和基准 asm 做差分** 。

```
# 来源：nv_isa_solver/instruction_solver.py（节选自 _analyse）
if self.parsed.get_key() != mutated_parsed.get_key():
    self.opcode_bits.add(i_bit); continue        # 签名变 → opcode
if self.parsed.predicate != mutated_parsed.predicate:
    self.predicate_bits.add(i_bit)               # 谓词变 → predicate 区
for i, (a, b) in enumerate(zip(mutated_operands, parsed_operands)):
    ifnot a.compare(b):
        self.operand_value_bits.add(i_bit)       # 操作数值变 → operand
        self.bit_to_operand[i_bit] = i
    else:
        effected, flag = analyse_modifiers(b.modifiers, a.modifiers)
        if effected:                              # 该操作数挂的 .modi 变了
            self.operand_modifier_bits.add(i_bit)
        if flag:                                  # 单独多了一个具名 token
            self.operand_modifier_bit_flag[i_bit] = flag
```

`analyse_modifiers` 是个不到 20 行的小函数，但它非常关键——它判断"修饰符的变化"到底是 **增减了某个具名标志（flag）** ，还是 **单纯修改了某个枚举字段（modifier）** 。这种区分让后续的可视化能把 `.S32` 、`.WIDE` 这样的助记符正确地标记为"独占位"。

### 3.3 多个分析 pass：修复一阶模糊的"误判"

> 单纯翻转一位有一个无法克服的盲区： **有些字段必须同时翻转两位才会触发变化** ，比如某些模糊器以为是 flag 的位其实只是一个 2-bit modifier 的低位。 `instruction_solver.py` 通过一组 fixed-point pass 来纠正：

- `analysis_disambiguate_flags` ：对每个"flag 候选位"，再翻转其相邻位，如果原 flag 助记符消失，就说明它其实是 modifier 的一部分。
- `analysis_extend_modifiers` ：在已识别的 modifier 字段两端各试探一位，看能否把字段扩长。
- `analysis_modifier_splitting` ：反向地，检查一个连续 modifier 字段是不是其实由两个 **互相独立** 的子字段拼成（用一个三段式 diff：原始 vs 改 A vs 改 A+B）。
- `analysis_operand_fix` ：处理像 `[UR10 + 0x1]` 这类"立即数为 0 时签名会变"导致的字段边界错位。

整套流程在主管道里被串成：

```
# 来源：nv_isa_solver/instruction_solver.py
def instruction_analysis_pipeline(inst, disassembler, arch_code):
    inst = disassembler.distill_instruction(inst)
    asm  = disassembler.disassemble(inst)
    mutations = disassembler.mutate_inst(inst, end=14 * 8 - 2)
    mset = InstructionMutationSet(inst, asm, mutations, disassembler)

    analysis_run_fixedpoint(disassembler, mset, analysis_disambiguate_flags)
    analysis_operand_fix(disassembler, mset)
    analysis_disambiguate_operand_flags(disassembler, mset)
    analysis_run_fixedpoint(disassembler, mset, analysis_extend_modifiers)
    analysis_run_fixedpoint(disassembler, mset, analysis_modifier_splitting)

    ranges = mset.compute_encoding_ranges()             # 把零散 bit 聚成字段
    modifier_values         = ranges.enumerate_modifiers(disassembler)
    operand_modifier_values = ranges.enumerate_operand_modifiers(disassembler)
    spec = InstructionSpec(asm, parsed_inst, ranges,
                           modifier_values, operand_modifier_values)
    spec.analyse_operand_interactions(arch_code, disassembler.nvdisasm)
    return spec
```

每个 pass 都是"幂等收敛"的：只要还能改进就重跑，直到 fixed-point。

## 四、字段重建：从比特集合到 EncodingRanges

> `compute_encoding_ranges` 把上一步的若干 `set[int]` 翻译成 **连续区间的列表** 。其逻辑像一个简单的状态机：遍历 0~127 每一位，按比特所属类型生成单位长度的 `EncodingRange` ，再判断能否与前一个区间合并：

```
# 来源：nv_isa_solver/instruction_solver.py（compute_encoding_ranges 精简）
for i in range(0, 8 * 16):
    new_range = None
    if i in self.modifier_bits:
        if i in self.instruction_modifier_bit_flag:
            _push()                                 # flag 是独立位，立即截断
            current_range = EncodingRange(
                EncodingRangeType.FLAG, i, 1,
                name=self.instruction_modifier_bit_flag[i])
            _push(); continue
        else:
            new_range = EncodingRange(
                EncodingRangeType.MODIFIER, i, 1,
                group_id=self.modifier_groups[i])
    elif i in self.predicate_bits:
        new_range = EncodingRange(EncodingRangeType.PREDICATE, i, 1)
    elif i in self.operand_value_bits:
        new_range = EncodingRange(EncodingRangeType.OPERAND, i, 1,
                                  operand_index=self.bit_to_operand[i])
    ...
    # 控制码区（第 13 字节后）按固定布局解释为 stall / yield / r-bar / w-bar / ...
    if (current_range and new_range.type == current_range.type
        and new_range.operand_index == current_range.operand_index
        and (new_range.group_id isNone
             or new_range.group_id == current_range.group_id)):
        current_range.length += 1                   # 合并到当前区间
    else:
        _push(); current_range = new_range
```

最终得到的 `EncodingRanges` 不仅可以反向 `encode(...)` 一条 **合法指令** （操作数寄存器号、修饰符值、predicate、stall、barrier 全部可指定），还能直接生成那张色彩斑斓的 HTML 比特表。 `encode` 函数是后续"活跃区间分析"和任何下游汇编器复用 ISA 规范的基石。

### 4.1 枚举每个字段的具体含义

> 光知道"第 24~27 位是 modifier"远远不够——我们还需要知道 `0b0001 = .S32` 、 `0b0010 = .U16` 之类的 **取值表** 。

`enumerate_modifiers` 干的就是这件事：固定其他位，把目标 modifier 字段从 0 枚举到 `2^len - 1` ，每个取值都喂给 `nvdisasm` ，然后用 `find_modifier_difference` 比较结果与基准 disasm 的助记符差异。这样每个 modifier 字段都得到一张"值 → 名字"的小表，最终渲染成 HTML 中的 Modifier Group 表格。

## 五、活跃区间分析：让每个操作数知道自己是"读"还是"写"

> 光有编码格式还不够，要真正写出可调度的代码生成器，还得知道 `IMAD R0, R1, R2, R3` 里哪个操作数被读、哪个被写。

`life_range.py` 利用了 `nvdisasm --print-life-ranges` 这个鲜为人知的开关——它会打印每个寄存器在指令窗口内的生命周期标记（ `^` 写、 `v` 读、 `x` 读写）。

但 `nvdisasm` 只接受合法的 cubin 文件，不接受裸字节。于是 `analyse_live_ranges` 借助仓库内嵌的 \`cubin\` <sup>[6]</sup> （移植自 TuringAs）模块 **当场合成一个最小可执行的 ELF cubin** ：把待分析指令 + 一条 EXIT 拼成 kernel，写到临时文件，再调用 nvdisasm 解析。

```
# 来源：nv_isa_solver/life_range.py
def analyse_live_ranges(inst, archCode=90, nvdisasm="nvdisasm"):
    bin = cubin.Cubin(arch=archCode)
    EXIT = bytes.fromhex("4d790000000000000000800300ea0f00")
    kernel = {
        "KernelData": inst + EXIT + b"\0" * 16 * 8,
        "ExitOffset": [], "BarCnt": 10, "RegCnt": 255, "SmemSize": 0,
    }
    tmp = tempfile.NamedTemporaryFile(delete=False); tmp.close()
    bin.add_kernel(kernel, b"test", {"name_list": [], "size_list": [0]},
                   {"name_list": [], "size_list": []})
    bin.Write(tmp.name)
    result = get_live_ranges(tmp.name, nvdisasm)
    os.remove(tmp.name)
    return result
```

而要让 `--print-life-ranges` 的输出 **正好把待测寄存器映射到我们关心的操作数槽位** ， `InstructionSpec.encode_for_life_range` 会为不同寄存器类型（R/UR/P/UP）分配间隔足够大的编号——比如 GPR 用 `16, 32, 48, …` ，UR 用 `4, 8, 12, …` ——这样每个操作数都落在一个独占的区间内，再去解析 nvdisasm 输出的表格就能精确反映出每个操作数的 READ/WRITE 性质。

## 六、把一切产出渲染成可读的 HTML 手册

> 最后一步是 `InstructionSpec.generate_html` ：每条指令生成一段 HTML，包括:

1. 助记符串带颜色高亮的操作数描述
2. 操作数的读写性质标签
3. 16 字节比特表（每位用不同颜色区分 opcode/operand/modifier/flag/控制码）
4. 每个 modifier group 的取值-名字对照表

运行完 `nv-isa-solver` 后， `output/` 目录下每个 base opcode 一个 HTML 文件，再加一个 `index.html` 总目录——这就是开篇提到的在线 SM90a 手册 <sup>[7]</sup> 的全部产物。

![英伟达GPU指令集里指令的一种编码格式，指令格式为，定义了1个谓词、5个寄存器操作数和1个特殊操作数，标注了各操作数的读写端口占用情况，还给出了指令的编码位域分布与修饰符组定义。](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

英伟达GPU指令集里指令的一种编码格式，指令格式为，定义了1个谓词、5个寄存器操作数和1个特殊操作数，标注了各操作数的读写端口占用情况，还给出了指令的编码位域分布与修饰符组定义。

## 七、为什么这件事重要：差分模糊测试的一次教科书式应用

**`nv_isa_solver` 的工程价值远超出"逆 NVIDIA"本身** 。它演示了一种通用方法论： ***当你面对一个不愿开放的二进制黑盒，但又能反复廉价地查询它时，差分模糊测试（differential fuzzing）就能把黑盒变成可消费的数据集** 。*

Zhang 等人 2017 年那篇 PPoPP 论文《Understanding the GPU Microarchitecture to Achieve Bare-Metal Performance Tuning》里的思路，在 Kuter 这个仓库里被实现得极其干净——蒸馏、翻转、归类、合并、枚举、字段重建，每个阶段都是 fixed-point 收敛，每个 pass 都和一个具体的"误判模式"对应。

对编译器后端开发者而言，它意味着：

- **不再依赖闭源 ptxas** ——拿到 `isa.json` 后，理论上可以自己写一个针对 H100 的汇编器/调度器；
- **可以审计 SASS 指令的副作用** ——通过活跃区间数据，自动构建依赖图与延迟模型；
- **可以快速跟进新架构** ——每出一代新卡，只要 CUDA Toolkit 里的 `nvdisasm` 能识别，就能在几小时内 re-run 整套流水线，得到一份新架构的初稿手册。

> 意义上 **， `nvdisasm` 才是 NVIDIA 真正"无意中开源"的 ISA 文档** ，而 `nv_isa_solver` 只是替我们按了一次"导出"按钮。下次当你抱怨 GPU 微架构资料匮乏时，不妨想一想：与其等厂商松口，不如读读这 1500 行 Python——你会发现， **逆向工作的优雅程度，可以比你想象中高得多** 。

参考资料

\[1\]

SM90a 在线版: *https://kuterdinel.com/nv\_isa/*

\[2\]

SM89 在线版: *https://kuterdinel.com/nv\_isa\_sm89/*

\[3\]

SM90a 在线版: *https://kuterdinel.com/nv\_isa/*

\[4\]

TuringAs: *https://github.com/daadaada/turingas*

\[5\]

CuAssembler: *https://github.com/cloudcores/CuAssembler*

\[6\]

`cubin`: *https://github.com/kuterd/nv\_isa\_solver/tree/main/nv\_isa\_solver/cubin*

\[7\]

在线 SM90a 手册: *https://kuterdinel.com/nv\_isa/*

相关推荐

- [暴涨 48 个 Star！探秘 NVidia SASS 逆向反汇编器 denvdis 的实现与微架构优化揭秘](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899353&idx=1&sn=d455cd54c0bf798ba8b9ac4a7cec5b24&scene=21#wechat_redirect)
- [逆向软硬件实现中的浮点累加顺序工具 FPRev](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447894524&idx=1&sn=0282673a4d170d4c5a14a563ad311d78&scene=21#wechat_redirect)
- [取代 NVIDIA 闭源 tileiras！开源编译器 FlashTile：一个透明、轻量、高效的 CUDA Tile IR 编译器！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447897055&idx=1&sn=eca7f075b981801fbb313158feb0945c&scene=21#wechat_redirect)

交流加群请在 NeuralTalk 公众号后台回复：加群

GPU · 目录

继续滑动看下一个

NeuralTalk

向上滑动看下一个