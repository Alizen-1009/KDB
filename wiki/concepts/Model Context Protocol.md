---
type: concept
topic: 推理服务
sources: 0
updated: 2026-08-17
---

# Model Context Protocol

## 全名与定义

`MCP` 全名是 **Model Context Protocol**，中文常译为“模型上下文协议”。它是一套开放协议，用统一方式把 AI 应用连接到外部数据、工具和可复用工作流。

人们口中的“MCP 服务”通常指 **MCP Server**：一个按照 MCP 协议向 AI 应用暴露能力的程序。这里的 `Server` 不一定是远程服务器；它也可以是由桌面应用在本机启动的子进程。

MCP 不是模型、数据库或 Agent 本身。它更像 AI 应用与外部系统之间的标准适配层。

## 它解决什么问题

### 统一集成接口

没有统一协议时，每个 AI 应用都要为 GitHub、数据库、文件系统、Slack 等分别编写私有插件；每个外部系统又要适配不同 AI 客户端，容易形成 `M × N` 的重复集成。

MCP 让外部系统按同一种方式描述能力，使支持 MCP 的 Host 能够发现和调用它们，降低重复适配成本。

### 动态能力发现

MCP Client 可以查询 Server 支持的协议版本、能力，以及当前可用的 tools、resources 和 prompts，而不必把所有能力硬编码进主程序。

### 结构化调用

Server 为工具声明名称、说明和输入 schema。Host 可以把这些信息交给模型，模型选择工具后，Client 用结构化 JSON-RPC 请求调用 Server，再把结果返回给 Host。

### 连接数据、动作与工作流

MCP 不只处理“调用函数”，还统一表达可读取的数据资源和可复用 prompt 模板，使 AI 应用能够在同一连接中获得上下文并执行动作。

### 复用与可移植性

一个 MCP Server 可以被多个支持 MCP 的 AI Host 使用；AI 应用也可以同时连接多个 Server，并由 Host 统一管理权限、交互与结果。

## 架构角色

```text
用户
  ↓
MCP Host（Claude Desktop、IDE、Coding Agent 等 AI 应用）
  ├── MCP Client A ── MCP Server A ── GitHub
  ├── MCP Client B ── MCP Server B ── 数据库
  └── MCP Client C ── MCP Server C ── 本地文件系统
```

- **MCP Host**：用户实际使用的 AI 应用，负责模型交互、权限、上下文和多个连接的协调。
- **MCP Client**：Host 内部的协议客户端；通常一个 Client 维护到一个 Server 的专用连接。
- **MCP Server**：按照 MCP 协议暴露能力的程序，并把请求转换成文件操作、数据库查询或第三方 API 调用。
- **外部系统**：真正保存数据或执行业务操作的 GitHub、Sentry、PostgreSQL、Slack、本地文件等。

因此“MCP Server”往往是一个适配器，而不是数据和业务本身。数据库仍然是数据库，GitHub API 仍然存在，MCP Server 只是把它们翻译成 AI Host 能统一发现和调用的接口。

## Server 暴露的三类核心能力

| Primitive | 含义 | 例子 | 主要控制方 |
| --- | --- | --- | --- |
| `Tools` | 可以执行的函数或动作 | 查询数据库、创建 issue、写文件 | 模型选择调用，Host 应负责授权与确认 |
| `Resources` | 可以读取的上下文数据 | 文件内容、数据库记录、Git 历史 | 应用选择如何获取和放入上下文 |
| `Prompts` | 可复用的交互模板 | 代码审查模板、事故分析流程 | 通常由用户显式选择 |

其中最常被讨论的是 `Tools`，但 MCP 不等于单纯的 Tool Calling。

## 一次工具调用如何发生

```text
1. Host 连接 MCP Server
2. Client 与 Server 初始化并协商版本、能力
3. Client 请求 tools/list
4. Server 返回工具名称、说明和输入 schema
5. Host 把可用工具告诉模型
6. 模型提出调用某个工具及参数
7. Host 做权限检查，必要时让用户确认
8. Client 发送 tools/call
9. Server 调用真实 API、数据库或本地程序
10. Server 返回结构化结果
11. Host 决定哪些结果进入模型上下文
```

模型通常不直接建立网络连接，也不应该直接持有全部凭证。Host、Client 与 Server 共同构成能力边界。

## 本地与远程 MCP Server

### 本地 Server：stdio

- Host 把 MCP Server 启动为本地子进程。
- 双方通过 `stdin/stdout` 交换 UTF-8 JSON-RPC 消息。
- 适合本地文件、Git、开发工具和桌面应用。
- Server 的日志应写到 `stderr`，不能污染承载协议消息的 `stdout`。

### 远程 Server：Streamable HTTP

- Server 作为独立网络服务，通过 HTTP 接收消息，并可用 SSE 提供流式响应或通知。
- 适合 SaaS、企业数据库、团队共享服务和云平台。
- 需要认证、HTTPS、Origin 校验、会话与多租户隔离；官方规范提供 OAuth 相关授权机制。

## 主要使用场景

### Coding Agent

让 Agent 读取仓库、查询 issue、调用构建系统、查看 Sentry 错误或操作云开发环境，而不用为每个 IDE 单独实现一套插件。

### 企业知识助手

把 Notion、Confluence、Google Drive、内部搜索和数据库以统一接口接入聊天助手，使模型按权限查询企业数据。

### 数据分析

Server 可以同时暴露数据库 schema 资源、SQL 查询工具和查询示例 prompt，帮助模型理解数据并执行分析。

### 运维与可观测性

连接 Sentry、日志平台、Kubernetes、监控系统和工单平台，让 Agent 查询告警、分析事故并在授权后执行操作。

### 个人助理

连接日历、邮件、任务管理和笔记系统，使助手读取日程、创建会议或整理任务。

### 专业软件控制

把 Figma、Blender、浏览器自动化或其他专业软件的动作封装成工具，让模型读取当前状态并执行操作。

## 与相近概念的区别

| 概念 | 主要解决的问题 | 与 MCP 的关系 |
| --- | --- | --- |
| REST/GraphQL API | 软件系统之间的业务接口 | MCP Server 经常在内部调用这些 API |
| Function/Tool Calling | 模型如何表达“我要调用某个函数” | MCP 进一步标准化工具发现、连接、执行和结果交换 |
| RAG | 从知识库检索文本并放入上下文 | MCP 可以暴露检索工具或资源，但不规定检索算法 |
| [[Sandbox]] | 限制代码可以访问什么、消耗多少资源 | MCP 暴露能力，Sandbox 隔离执行；两者经常组合 |
| [[LLM Programs]] | 组织多步模型调用、工具和控制流 | MCP 可以作为 LLM Program 的标准工具与数据接口 |
| 插件系统 | 给单个应用安装扩展 | MCP 更强调跨 Host、跨服务的开放协议和互操作性 |

## MCP 与 Sandbox 的关系

两者解决不同问题：

```text
MCP：Agent 能调用哪些外部能力，以及怎样调用
Sandbox：Agent 执行代码时被限制在什么边界内
```

例如 Coding Agent 可以通过 MCP 获得 `create_github_issue` 工具，同时把本地测试命令放进 Sandbox 执行。MCP 不会自动把工具调用变安全，Sandbox 也不会自动提供 GitHub、数据库等业务能力。

## 安全边界

- MCP Server 暴露的工具可能读文件、发消息、改数据库甚至部署生产环境，因此不能因为“使用标准协议”就默认可信。
- Host 应展示工具来源、参数和敏感动作，并对高风险调用请求用户确认。
- Server 应使用最小权限凭证、细粒度 scope、短期 token 和独立审计日志。
- 远程服务需要 HTTPS、正确认证和 token audience 校验；本地服务应避免监听所有网卡，并防范 DNS rebinding。
- 工具返回内容也可能包含 prompt injection，Host 不应把外部内容无条件当成可信指令。
- 安装第三方 MCP Server 等价于安装拥有一定本机或账号权限的软件，应检查来源和实际权限。

## MCP 不能解决什么

- 不会自动让模型理解所有工具，也不保证模型会选择正确工具和参数。
- 不会替代权限系统、审批、审计、Sandbox 和业务侧校验。
- 不会取代原有 API；Server 通常只是现有系统的 MCP 适配层。
- 不保证不同 Server 的业务语义一致；协议统一不等于工具设计质量统一。
- 对只有一个固定客户端和一个简单内部 API 的系统，直接集成可能比引入 MCP 更简单。

## 相关概念

- [[LLM Programs]]
- [[Sandbox]]
- [[Recursive Language Model]]
- [[Context Folding]]

## 官方资料

- [What is the Model Context Protocol?](https://modelcontextprotocol.io/docs/getting-started/intro)
- [Architecture overview](https://modelcontextprotocol.io/docs/learn/architecture)
- [Server concepts](https://modelcontextprotocol.io/specification/2025-06-18/server/index)
- [Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)

## 研究备注

- MCP 规范持续演进，具体生命周期、认证和 client-side primitives 应按部署所用协议版本核对。
- 评估 MCP Server 时应同时检查工具 schema 质量、权限范围、错误语义、幂等性、超时、审计和 prompt injection 风险，而不只看工具数量。
