# Scripts

这些脚本对应 LLM Wiki 的三个核心动作：ingest、query、lint。

## `ingest.py`

为新原始资料创建摄入任务，并把本次操作写入 `wiki/log.md`。

这个任务模板采用“先讨论、再落盘”的两阶段流程：

1. 先读完整原文
2. 先向你汇报 `2-3` 条摘要和值得关注的论断
3. 等你确认后，再更新 `wiki/sources/`、`wiki/entities/`、`wiki/concepts/`

```bash
python3 scripts/ingest.py raw/papers/flashattention-3.pdf
python3 scripts/ingest.py raw/repos/vllm/README.md
```

## `query.py`

把研究问题转为结构化输出文档，默认写入 `output/reports/`。

```bash
python3 scripts/query.py "对比 vLLM、SGLang 和 TensorRT-LLM 的调度设计"
python3 scripts/query.py "FSDP 与 ZeRO 在训练侧的权衡" --format slides
```

## `lint.py`

对 wiki 做健康检查。

```bash
python3 scripts/lint.py
```

## `update_index.py`

扫描知识库并重建 `wiki/index.md`。

索引页会统计 `sources / entities / concepts` 三类核心 wiki 页面。

```bash
python3 scripts/update_index.py
```

## `health_check.py`

`lint.py` 的底层实现文件，可单独运行。
