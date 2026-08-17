---
type: concept
topic: 推理服务
sources: 2
updated: 2026-08-17
---

# Sandbox

## 定义

`Sandbox`（沙箱）是一种**受限制、可隔离、通常可销毁的执行环境**：允许程序运行代码、命令和工具，但限制它能访问的文件、进程、网络、凭证与计算资源，并尽量避免影响宿主机和其他用户。

“沙箱”不是某个固定产品，也不是绝对安全等级。浏览器、容器、gVisor、Wasm 和 microVM 都可以提供不同强度的沙箱边界。

## 它解决什么问题

- **安全隔离**：不让不可信代码读取宿主机文件、攻击内网或控制其他进程。
- **多租户隔离**：不同用户、Agent 和任务之间不能互相读取或修改状态。
- **资源控制**：限制 CPU、内存、进程数、磁盘、运行时间和网络流量，防止死循环、fork bomb 或写爆磁盘。
- **环境复现**：固定操作系统、语言 runtime 和依赖版本，减少“在我机器上能跑”的环境差异。
- **生命周期管理**：任务结束后销毁 writable layer 和临时凭证，避免污染后续任务。
- **审计与回收**：记录执行、文件和网络行为，并对超时、异常或空闲环境进行终止和清理。

## 关键隔离维度

| 维度 | 需要限制的内容 |
| --- | --- |
| 文件系统 | 只挂载任务 workspace；rootfs 可只读；禁止宿主敏感路径 |
| 进程与系统调用 | 隔离 PID，drop capabilities，限制或拦截危险 syscall |
| 计算资源 | CPU、内存、PIDs、I/O、磁盘、GPU、执行时间配额 |
| 网络 | 默认禁止或按 allowlist 放行外网；阻止内网与 metadata service 探测 |
| 身份与凭证 | 只注入任务所需的短期、最小权限 token，结束后立即失效 |
| 状态与生命周期 | 每会话独立 writable layer；结束后销毁、归档或按策略恢复 |
| 可观测性 | 记录 exec、文件修改、网络连接、资源峰值和退出原因 |

## 常见实现层级

| 方案 | 隔离边界 | 特点与典型场景 |
| --- | --- | --- |
| 语言/runtime 限制 | 语言解释器或字节码 runtime | 启动快、能力有限；如受控表达式、Wasm/WASI |
| Linux 容器 | namespace、cgroup、seccomp、capabilities | 性能与启动速度好，但仍共享宿主机 kernel |
| gVisor | 用户态 kernel / syscall 拦截层 | 减少宿主 syscall 暴露，兼容性和性能有额外代价 |
| Kata / Firecracker microVM | 独立 guest kernel / 虚拟机边界 | 隔离更强，适合多租户不可信代码；启动和运维更复杂 |
| 独立远程 VM | 完整机器或云实例边界 | 隔离与资源灵活性高，但成本和冷启动通常更高 |

## Docker 与 Sandbox 的区别

Docker 容器可以是沙箱实现的一部分，但“放进 Docker”不等于已经安全：

- 容器共享宿主 kernel，kernel 漏洞仍可能形成逃逸路径。
- 挂载 Docker socket、宿主目录或使用 privileged container，几乎会破坏隔离。
- 未限制 cgroup、网络和磁盘时，代码仍可能耗尽宿主资源或攻击内网。
- 长期凭证如果直接注入容器，恶意代码仍可读取并外传。

因此是否需要普通容器、gVisor 还是 microVM，应由威胁模型决定，而不是只看启动速度。

## Agent 场景中的 Sandbox

AI Agent 会生成并执行 Python、Shell、包安装、文件修改和网络请求。即使用户本身可信，模型也可能因为幻觉、错误命令或网页 prompt injection 执行危险操作。典型执行链是：

```text
Agent 请求执行代码
  → Sandbox Scheduler 选择隔离等级与镜像
  → 创建独立 workspace / network policy / resource quota
  → 在 sandbox 内执行并收集 stdout、文件和退出状态
  → 超时终止或返回结果
  → 销毁 writable layer，或保存会话状态供后续恢复
```

[[Recursive Language Model]] 中的持久 Python REPL 就适合运行在沙箱里：模型可以安装依赖和处理外部数据，但不应因此获得宿主机或其他会话的权限。

## 常见使用场景

- Coding Agent、RLM、Notebook 和数据分析助手执行模型生成的代码。
- 在线判题、代码面试平台和教育网站运行用户提交程序。
- CI/CD 执行来自分支或 Pull Request 的构建与测试。
- SaaS 平台运行第三方插件、脚本和用户自定义函数。
- 浏览器隔离网页 JavaScript 与本地系统。
- 安全研究中动态分析恶意文件或未知程序。
- Agentic RL 中批量启动可重置、可评分的交互环境。

## 性能与工程权衡

- 隔离越强，通常冷启动、内存开销、系统调用兼容性和运维复杂度越高。
- 秒级 Agent sandbox 常通过预拉镜像、warm pool、快照恢复、overlayfs 和状态外置降低启动时间，而不是每次冷启动完整虚拟机。
- 持久会话体验更好，但增加状态泄漏、资源占用和清理复杂度；一次性环境更安全易回收，但重复初始化成本更高。
- GPU sandbox 还要处理设备直通、驱动攻击面、显存残留、MIG/时间片隔离和调度，不能只复用 CPU 容器方案。

## 不能解决什么

- Sandbox 只能降低风险，不能保证不存在 kernel、hypervisor 或 runtime 漏洞。
- 它不能替代应用层权限校验、输入输出验证和供应链安全。
- 如果主动授予了敏感目录、宿主 socket、内网访问或长期密钥，沙箱无法补救错误的权限设计。
- 它也不能自动判断模型生成的业务操作是否正确，例如误删沙箱内合法工作区数据。

## 相关实体

- [[../entities/Prime Intellect]]
- [[../entities/verifiers]]

## 相关来源

- [[../sources/Recursive Language Models the paradigm of 2026]]
- [[../sources/量化剪枝推理瓶颈Nsight与异构集群面试整理]]

## 相关概念

- [[Recursive Language Model]]
- [[LLM Programs]]

## 研究备注

- 设计前应先写清楚威胁模型：代码由谁提供、允许哪些网络和文件访问、是否多租户、是否需要 GPU、逃逸后的最大损失是什么。
- 后续可补 Prime Intellect Sandboxes 官方文档以及 gVisor、Firecracker、Kata Containers 的一手架构资料。
