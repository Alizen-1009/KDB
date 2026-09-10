# vLLM、TGI、TensorRT-LLM、SGLang 如何比较？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- vLLM、TGI、TensorRT-LLM、SGLang 与题单中的 ImDeploy 有何核心区别？
- vLLM、TGI、TensorRT-LLM、ImDeploy、SGLang 在推理加速上的核心区别
- vLLM vs TGI vs TensorRT-LLM vs SGLang

## 30 秒回答

原稿把 vLLM 概括为分页缓存与连续批处理驱动的通用 serving，TGI 偏 Hugging Face 生态与部署集成，TensorRT-LLM 偏 NVIDIA 深度优化，SGLang 偏前缀复用与程序化推理；部署工具链与 LLM serving engine 不能直接等同。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/entities/TensorRT-LLM|TensorRT-LLM]]
- [[../../../../wiki/entities/vLLM|vLLM]]
- [[../../../../wiki/entities/SGLang|SGLang]]

## 参考来源与待核实

- [[../../大模型系统面试题全答#5. vLLM、TGI、TensorRT-LLM、ImDeploy、SGLang 在推理加速上的核心区别|大模型系统面试题全答]]
- [[../../大模型系统面试题地图#3. 推理系统与服务编排|大模型系统面试题地图]]

- 待核实 / 原稿边界：重大命名风险：题干写 ImDeploy，答案改称 MMDeploy/ImDeploy，未说明是否原意为 LMDeploy，不可静默纠正。框架定位为粗粒度、无版本的旧口径，当前功能对比待核实。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
