---
title: "Graph Engineering with Claude 中文翻译"
source: "https://x.com/0xCodez/status/2079165300625330317"
author: "Codez (@0xCodez)"
published: 2026-07-20
translation_status: "全文翻译"
---

# 使用 Claude 进行图工程：从零到图架构师的 14 步路线图（完整教程）

> [!note] 译者说明
> 本文为 `raw/articles/Graph Engineering with Claude.md` 的中文翻译。为便于核对，保留了 `node`、`edge`、`fan-out`、`barrier`、`pipeline` 等关键英文术语和原始代码。文中关于 Claude Code 具体功能与 API 行为的描述均属于原作者说法，尚未通过官方文档独立核实。

大多数人在尝试构建多步骤 Agent 时，最后做出来的都是一条直线：第一步、第二步、第三步——每一步都礼貌地等待上一步完成，然后才开始运行。

**十分之九的人没有意识到，其中一半的步骤根本不需要等待。**

这些工作流不会**路由（route）**，不会**分支（branch）**，也不会**并行化（parallelize）**。它们只会排队：一个主脑、一个上下文、一次只做一件事，直到上下文窗口被塞满，Agent 忘记自己原本在做什么。

> 关注我的 Substack，获取最新 AI 动态：[movez.substack.com](https://movez.substack.com/)

这是一条包含 **14 个步骤的路线图**，用于把单文件中的线性流程转变成一张**图（graph）**：它可以把工作并行分发给一组 Agent，自动验证自己的发现，并最终收敛到一个单独 Agent 无法独自容纳和完成的结果。

![](https://pbs.twimg.com/media/HNqdriNXYAAJlBB.jpg)

这里有一个很少有人明确说明的认知转变：prompt 是一句话，loop 是一个循环，harness 是 Agent 脚下的运行底座。

但**工作本身的形状**——哪些任务必须先运行，哪些任务可以同时运行，哪些任务必须等待其他所有任务——这个形状其实是一张图。节点负责思考，边负责传递结果。

Claude Code 已经提供了直接构建这些图的工具：**动态工作流（dynamic workflows）**。

Claude 会编写一个普通的 JavaScript 编排脚本，然后启动一组相互协作的子 Agent 来执行它。因为协调逻辑是代码而不是对话，所以协调本身不消耗模型 token。

---

## 01. 节点是任务，边是流动的数据

一张图恰好包含两类东西。把它们区分清楚，就能消除大多数困惑。

一个**节点（node）**是一个工作单元：一个 Agent、一个边界明确的任务、一个输入和一个输出。

一条**边（edge）**是一项依赖关系：它表示“这个节点的输出会成为另一个节点的输入”。仅此而已。

![](https://pbs.twimg.com/media/HNqlJ68XUAAe-PB.png)

常见错误是把“然后（and then）”当成一条边。例如：“总结这个文件，**然后**告诉我天气。”这两个任务之间其实没有边，因为查询天气并不会消费文件摘要。

它们是两个互不相连的节点，只是被线性脚本不必要地串在了一起。只有当数据确实从一个节点流向另一个节点时，边才真正存在。

对于 Agent 工作流里的每一个“然后”，都要问自己：**下一步是否会读取上一步的输出？** 如果不会，就不存在边，等待也只是浪费。

```text
把工作流画成方框和箭头。一个方框代表一次 agent() 调用。
一条箭头代表某次调用的返回值被作为变量传进另一次调用的 prompt。

如果你画不出这条箭头——也就是说，没有变量跨越两个方框——那么这两个
方框就是相互独立的。接下来整门课程所要利用的，正是这种独立性。
```

---

## 02. 线性脚本是一种退化的图

当你把 Agent 写成“执行 A，然后 B，然后 C，然后 D”时，你其实已经画出了一张图——一条没有任何分支的链。每个节点都恰好有一条入边和一条出边。

它可以正确运行，但运行得很慢，也很脆弱，因为链条没有冗余：如果 C 卡住，D 就永远不会执行；A 的成果也被困在上游，无处可去。

![](https://pbs.twimg.com/media/HNql8XpXcAA5-2h.png)

图工程的第一个真正技能是**重新绘制这条链**。拿出你的线性 Agent，对每一条箭头都问一遍第 1 步的问题。

大多数链条里都有两三条箭头并不携带数据，只是因为你碰巧按那个顺序写了代码，所以它们才存在。

剪掉这些箭头，链条就会坍缩成一种更宽的结构：若干相互独立、可以同时运行的节点，最终共同汇入一个确实需要所有结果的节点。

---

## 03. 为每个节点定义契约

如果你无法推理一个节点的行为，就无法安全地将它并行化。解决方法是为节点定义契约：**有边界的输入、有边界的输出，并且只承担一个任务。**

输入是节点读取的全部内容——必须显式传入，不能默认它存在于某个共享上下文窗口中。输出应当具有确定的结构，最好经过验证，使下游节点无需猜测就能直接消费。

![](https://pbs.twimg.com/media/HNqmfUhXIAA_Y70.jpg)

在工作流中，这类契约通过 **schema** 强制执行。当你给 Claude 的 `agent()` 调用附上 JSON Schema 时，Claude 启动的子 Agent 就必须返回通过验证的结构化数据。验证发生在工具调用层；如果结果不匹配，Claude 会重试，而不是交给你一段需要自己解析并祈祷格式正确的自由文本。

这正是“可以被 Claude 接入图中的节点”和“只有人类阅读输出时才能工作的节点”之间的区别。

```javascript
// 一个具有真正契约的节点：输入有边界、输出经过验证、只负责一项工作。
const ITEM = {
  type: 'object', additionalProperties: false,
  properties: {
    title:   { type: 'string' },
    url:     { type: 'string' },
    impact:  { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['title', 'url', 'impact'],
};

const result = await agent(source.prompt, {
  label:  `research:${source.key}`,
  schema: ITEM,           // 强制返回经过验证的结构化输出
  agentType: 'general-purpose',
});
// 此时 result 已经具有下游节点可以信任的结构，而不是自由文本。
```

---

## 04. 把边视为数据契约

边并不只是“B 在 A 之后执行”。它是一项关于所传数据的承诺：A 生成**这种结构**，而 B 被设计为消费**这种结构**。

如果用数据而不是执行顺序来命名一条边，两件事会变得更容易。

![](https://pbs.twimg.com/media/HNqnSStWIAAPvab.png)

第一，你能立即看出这条边是否真实存在——数据是否真的发生了移动？第二，只要数据结构保持不变，你就可以替换边任意一端的节点，而不会破坏整张图。

在实践中，边存在于普通 JavaScript 代码中。fan-out 和综合节点之间的 reduce 步骤——例如展开、去重、过滤——只是对节点返回结构执行操作的代码。

**这里不需要 Agent。** 图思维带来的一个隐性收益是：人们大量消耗模型 token 去做的工作，其实只是边上的数据处理，而边可以免费执行。

```text
你很容易产生一种冲动：再启动一个 Agent 来“合并结果”。请克制这种冲动。

如果所谓合并只是展开列表并去重，那么使用 results.flatMap(...) 和一个 Set
就够了——行为确定、立即完成、消耗零 token。把 Agent 留给需要判断的工作，
不要让它承担管道连接工作。

如果一张图里的每条边都是一个 Agent，那么这张图就是在为自己的线路支付租金。
```

---

## 05. 使用 `parallel()` 扇出

这是能让此前所有设计获得回报的关键动作。当你有 N 个相互独立的节点——需要检查 N 个来源、审查 N 个文件或审计 N 条路由——不要把它们串成一条链。

让 Claude 把它们扇出并同时执行。在工作流中，这由 `parallel()` 完成：Claude 接收一个 thunk 数组，为每个 thunk 启动一个子 Agent，并发执行它们，最后返回结果数组。

![](https://pbs.twimg.com/media/HNqn3q8XoAAXCdJ.jpg)

有两个细节能让这个过程更稳健。

第一，`parallel()` 是一道**屏障（barrier）**：它会等待所有 thunk 完成后再返回，因此下一阶段能看到完整结果集。

第二，某个抛出异常的 thunk 会解析为 `null`，而不是导致整个批次被拒绝。因此，一个不稳定的 Agent 不会拖垮整次运行。

务必对结果调用 `.filter(Boolean)`。并发数大约受 CPU 核心数量限制，超出的任务会排队。因此，即使传入一百个 thunk，它们最终也都会完成，只是每次同时运行其中一小部分。

```javascript
phase('Research');

// 九个来源、九个 Agent，同时运行。
const raw = await parallel(
  SOURCES.map((s) => () =>
    agent(s.prompt, {
      label: `research:${s.key}`,
      phase: 'Research',
      schema: ITEM_SCHEMA,     // 每个节点返回经过验证的 JSON
      agentType: 'general-purpose',
    }),
  ),
);

const collected = raw.filter(Boolean);  // 丢弃失败 Agent 产生的 null
```

fan-out 存在于 Claude 编写的代码中，而不是模型对话中。Claude 自己的上下文从未同时容纳九个来源——每个子 Agent 都有自己的上下文，只有最终答案会返回。

正因为如此，Claude 才能把一个工作流扩展到**几十甚至几百个子 Agent**，而不会淹没主会话。编排层不消耗 token，因为它不是 Claude 的另一轮思考。

---

## 06. 在屏障处扇入

如果没有任何东西收集结果，fan-out 就没有价值。**扇入（fan-in）**是多条边汇聚的节点：一个 Agent 或一段代码在这里同时看到**所有**上游结果，并执行确实需要完整结果集的工作，例如跨来源去重、按影响力排序，或者在所有结果都为空时提前退出。

这是少数值得付出屏障等待时间的地方。

![](https://pbs.twimg.com/media/HNqoqYPWcAEYbIc.jpg)

保持图高效的规则是：**只有当一个阶段确实需要此前所有结果时，才使用屏障。** 对所有来源进行统一去重？需要屏障，这是正确的。

```javascript
// 边：普通 JavaScript，不需要 Agent，消耗零 token。
const flat = collected.flatMap((c) => c.items);
log(`Collected ${flat.length} items`);

phase('Curate');
// 屏障节点：需要完整结果集才能进行去重和排序。
const curated = await agent(
  `Dedupe and rank these by impact:\n${JSON.stringify(flat)}`,
  { phase: 'Curate', schema: CURATED_SCHEMA },
);
```

只是把列表展开？那只是边上的处理，直接内联执行即可。

判断方法简单而严格：如果你写出了 `parallel → transform → parallel`，而中间的 transform 不存在跨条目依赖，那么你本来应该使用 pipeline，并完全跳过这道屏障。

---

## 07. 菱形结构：拆分 → 工作 → 合并

把 fan-out 和 fan-in 组合起来，就得到了所有严肃 Agent 图中最常用的拓扑：**菱形（diamond）**。

一个节点拆分任务，许多节点并行执行工作，最后一个节点合并结果。这是市场扫描、依赖审计、代码审查和研究报告背后的共同形状。只要替换数据源和 prompt，同一个骨架就能适应不同任务。

![](https://pbs.twimg.com/media/HNqpB0FWgAAM74x.png)

它的标准形式值得记住：**fan out → reduce → synthesize（扇出 → 规约 → 综合）**。

通过 fan-out 获得广度；用普通代码进行 reduce，以压缩结果；最后使用一个 Agent 综合并写出答案。

一旦看见菱形结构，你就不会再问“怎样让我的 Agent 执行更多步骤”，而会开始问“在哪里拆分、在哪里合并”。后一个问题才是真正能够扩展的问题。

---

## 08. 使用条件判断在运行时路由边

并非所有图都是固定的。有时应该选择哪条边，取决于某个节点发现了什么。

**路由节点（router node）**检查结果，并决定触发哪条下游路径。例如，先对工单分类，再分支到正确的处理器；先检查 diff 大小，再决定做一次快速审查，还是启动完整审计。

在工作流中，这只是 JavaScript 根据节点的已验证输出执行一次 `if` 或 `switch`，因为控制流存在于代码中。

![](https://pbs.twimg.com/media/HNqpYomX0AAPw7z.png)

此时，确定性不再是一项限制，而是一项能力。路由器的**决策**可以由 Claude 提供——例如让子 Agent 进行分类——但实际的**路由动作**由 Claude 编写的脚本执行。对于同样的分类结果，它每次都会以相同方式运行。

你同时得到了节点中的 Claude 判断能力和边上的脚本可靠性。不会再出现“Claude 突然决定跳过审计”这种不可预测的意外，因为除非图中明确写了跳过逻辑，否则它就无法跳过。

```javascript
// 路由节点：Agent 负责分类，代码负责选择边。
const { severity } = await agent(
  `Classify this diff's risk:\n${diff}`,
  { schema: { type: 'object',
      properties: { severity: { enum: ['low', 'high'] } },
      required: ['severity'] } },
);

let review;
if (severity === 'high') {
  // 重路径：完整的并行审计
  review = await parallel(FILES.map((f) => () => agent(`Audit ${f}`)));
} else {
  // 轻路径：一次快速审查
  review = await agent(`Quick review of ${diff}`);
}
```

---

## 09. 在边上放置验证器

图真正的杠杆并不是更多 Agent，而是你能围绕这些 Agent 构建怎样的结构，从而获得可信度。

**验证器节点（verifier node）**位于一条边上。在一个结果被允许流向下游之前，它只负责做一件事：尝试**推翻这个发现**。如果结果经受住了挑战，它就可以通过；否则，它永远不会进入最终答案。

![](https://pbs.twimg.com/media/HNqp1zkW8AA4fhg.png)

有三种模式值得掌握：

- **对抗式验证（adversarial verify）：**针对每条发现，启动 N 个相互独立、以反驳它为目标的怀疑者；只有多数验证者未能推翻它时，才保留该发现。
- **多视角验证（perspective-diverse verify）：**给每个验证器分配不同视角，例如正确性、安全性、能否复现。多样化检查可以发现 N 个同质检查永远捕捉不到的失败模式。
- **评审委员会（judge panel）：**从不同角度生成 N 份候选答案，让多个评委并行评分；以获胜答案为基础进行综合，同时吸收其他候选中的优秀部分。

据文中所述，某个真实团队正是利用这种模式，将对抗式代码审查嵌入循环，从而完成了 Bun runtime 的移植。

---

## 10. 隔离节点，避免单点故障污染整张图

在一条链中，故障会级联：C 失败，D 就永远不会运行，整个流程停止。在一张图中，故障应当被**限制在发生故障的节点内**。

这在一定程度上已经成立：`parallel()` 中某个抛出异常的 thunk 会返回 `null`，因此即使一个 Agent 失败，另外八个正常 Agent 仍能返回结果。`.filter(Boolean)` 就是故障隔离机制的一部分。

每个 fan-in 都应被设计成能够容忍输入缺失，而不是默认一定会收到完整结果集。

![](https://pbs.twimg.com/media/HNqqdlGXcAAYFV3.png)

还有一种更隐蔽的故障：节点之间相互踩踏。当多个 Agent 并行写文件时，它们可能发生冲突。

解决方法是使用 `worktree` 隔离：每个 Agent 都在自己的 Git worktree 中运行，在独立沙箱里完成工作，然后再干净地合并。

只有当节点确实需要并行写入文件时才使用它。它是为这种拓扑准备的安全带，而不是每次运行都必须承担的默认成本。

---

## 11. 加入循环，但必须保证它收敛

有时，在真正开始工作前，你并不知道任务规模有多大。例如，规模未知的探索任务，或者一个 bug 扫描：找到一个 bug 后，又暴露出三个新 bug。

这种任务需要一个**循环（cycle）**——一条受控地返回早期节点的边。

危险很明显：无法收敛的循环就是无限循环，它会不断启动 Agent，直到预算耗尽。

![](https://pbs.twimg.com/media/HNqrFflXAAEGQxk.png)

能够收敛的模式叫作 **loop-until-dry（循环直到不再产生新结果）**：持续启动查找器，直到连续 K 轮都没有发现任何新内容，然后停止。

其中有一个决定成败的细节，也是几乎所有人第一次实现时都会犯的错误：应该针对什么集合进行去重？

答案是：针对所有**已经见过（seen）**的内容去重，而不只是针对已经确认（confirmed）的结果。

否则，被验证器拒绝的发现会在每轮中重新出现，循环永远无法进入“无新发现”状态。你最终构建的是一台不断花钱重复发现相同死胡同的机器。

```javascript
const seen = new Set(); const confirmed = []; let dry = 0;

while (dry < 2) {                       // 连续 2 轮为空后停止
  const found = (await parallel(
    FINDERS.map((f) => () => agent(f.prompt, { schema: BUGS }))
  )).filter(Boolean).flatMap((r) => r.bugs);

  const fresh = found.filter((b) => !seen.has(key(b)));
  if (!fresh.length) { dry++; continue; } // 没有新内容 → 接近停止
  dry = 0;
  fresh.forEach((b) => seen.add(key(b))); // 针对 SEEN 去重，而非 confirmed

  // 使用不同视角验证每个新发现，验证通过后才计入结果
  const judged = await parallel(fresh.map((b) => () =>
    parallel(['correctness', 'security', 'repro'].map((lens) => () =>
      agent(`Judge "${b.desc}" via ${lens} — real?`, { schema: VERDICT })))
    .then((v) => ({ b, real: v.filter(Boolean).filter((x) => x.real).length >= 2 }))));

  confirmed.push(...judged.filter((v) => v.real).map((v) => v.b));
}
```

---

## 12. 在不同节点之间分配不同档位的模型

不是每个节点都需要使用最强的模型。一张图能让这一点变得非常明显，而单 Agent 往往无法做到。

有些节点边界明确且高度重复，例如提取某个字段、对工单分类；另一些节点才真正承担关键判断，例如综合报告、裁决一条发现。

让便宜的模型运行枯燥的节点，把昂贵的 token 花在真正需要判断力的地方。

![](https://pbs.twimg.com/media/HNqrj4UWYAAvHC0.jpg)

在一个工作流中，Claude 启动的每个子 Agent 默认会继承当前会话的模型，除非脚本显式覆盖。因此，一次大型运行默认会全部按当前会话的模型档位计费。

单次 `agent()` 调用中的 `model` 选项可以让 Claude 只把该节点路由给其他模型。

**在大型运行前检查 `/model`。** 随后，让 Claude 把 fan-out 中重复、机械的节点下放到便宜模型，把合并节点保留在更强模型上。

这样无需改变图的结构，就能把一个消耗大量 token 的昂贵工作流变成更经济的工作流。

---

## 13. 拓扑结构决定成本和延迟

图的形状并不只是装饰——它是影响端到端时间最重要的杠杆。

最容易让人犯错的选择是：应该使用 `parallel()` 还是 `pipeline()`？

`parallel()` 的屏障会强制**所有任务**等待最慢的节点完成，然后下一阶段才能开始。

`pipeline()` 则让每个条目独立地流过所有阶段，不设置屏障。条目 A 可能已经进入第 3 阶段，而条目 B 仍停留在第 1 阶段。较快的条目可以提前完成，而不必因为较慢的条目空等。

![](https://pbs.twimg.com/media/HNqr10sW0AAFINc.png)

**默认使用 `pipeline()`。** 只有当某个阶段真正需要此前的全部结果时，才使用屏障，例如：

- 需要对整个结果集进行去重；
- 需要根据结果总量提前退出；
- prompt 必须把当前发现与“其他发现”进行比较。

“这样代码看起来更整洁”或者“这些阶段感觉彼此分开”都不是设置屏障的理由。屏障造成的延迟是真实、可测量且被浪费掉的时间。

彼此分离不等于必须同步。

---

## 14. 让 Claude 自己绘制图：自路由

最后一步，是对于那些无法提前规划的工作，不再手工绘制图。

借助**动态工作流（dynamic workflows）**，你只需要描述目标，Claude 会自行编写编排脚本：分解任务、选择 fan-out 方式、启动一组协调运行的子 Agent，并综合最终结果。

你得到的是专门适配**本次运行**的图，而不是一张预先固定、只能希望它恰好合适的图。

![](https://pbs.twimg.com/media/HNqsMKpXMAAB09_.jpg)

文中给出了三种入口：

1. 在 prompt 中使用 **“workflow”** 一词，Claude 就会为该任务编写工作流。
2. 运行一个已保存或内置的工作流。例如，`/deep-research` 是一张已经投入使用的图：界定范围 → 并行搜索 → 获取资料 → 对抗式验证 → 综合。这正是本文介绍的骨架。
3. 开启 `ultracode`，让 Claude 为会话中的每个重大任务规划工作流。

如果某次运行效果很好，可以按 `s` 把脚本保存到 `.claude/workflows/`。这样它就会进入版本控制，可以按名称重复运行；任何克隆该仓库的人都能启动它。

```text
› 运行一个 workflow，审计 src/routes/ 下的每条路由，检查是否缺少认证。
  为每个路由文件启动一个 Agent，并在报告前验证每条发现。

● Claude 编写了编排脚本 · 正在后台启动……

/workflows — auth-audit · running
✓ Scope       1/1   2.1k tok · 4s
✓ Fan-out    18/18  每个路由文件一个 Agent
◯ Verify     11/18  每条发现由 3 个怀疑者投票
○ Synthesize  0/1   等待验证完成

会话仍可响应——当 Agent 集群运行时，你可以继续工作。
```

---

## 本周可以用 Claude 构建的六种图

![](https://pbs.twimg.com/media/HNqtS-UXkAAKzv.png)

### 1. 对所有路由执行安全扫描

Claude 为**每个路由文件启动一个子 Agent**，分别寻找缺失的认证检查；随后执行验证器阶段，确保每条发现经过确认后才进入报告。

这种广度超出了单个上下文能够容纳的范围。

### 2. 使用 `/deep-research` 生成带引用的报告

这是一张已经随 Claude Code 提供的图。Claude 把问题分解为不同角度，并行搜索，对来源去重，然后使用三个投票式怀疑者**对抗性地验证每项论断**，最后撰写报告。

### 3. 逐文件移植模块

把 Bun 的案例扩展到你的仓库。Claude 把文件翻译工作并行扇出，针对每个结果运行测试套件作为门禁，并把失败项送回循环；通过**对抗式审查**发现单次执行可能直接交付出去的错误。

### 4. 对 diff 进行对抗式审查

Claude 根据 diff 大小进行路由：小改动只接受一次快速审查，大改动则触发**完整并行审计**。

多个审查者从正确性、安全性、性能等不同视角检查代码，随后由评审委员会综合结果。

### 5. 定期执行生态扫描

保存一次，永久重复运行。Claude 并行检查多个来源——版本发布、博客和讨论——在一道屏障处按影响力排序，然后编写摘要。

工作流在 `.claude/workflows/` 中接受**版本控制**，并且可以按名称启动。

### 6. 探索规模未知的问题

你不知道系统中有多少 bug。Claude 并行运行多个查找器，把每条新发现与**所有已经见过的内容**去重，验证剩余候选，并持续循环。

当连续两轮都没有发现新内容时，循环停止。

---

## 结论

使用 prompt 的人提出问题，架构师绘制一张**图**。

线性 Agent 从来都不是能力上限——它只是人们最先想到的一种形状，因为它与我们打字的方式一致：**一条线、一个主脑、一次只做一件事。**

一旦你能够看见节点和边，就不会再要求 Agent 做更多，而会开始要求图把工作做得更宽：

- 在工作相互独立的地方 fan-out；
- 在结果需要可信度的地方为边设置验证门禁；
- 在不需要高级判断的地方使用更便宜的模型。

大多数人仍会继续把步骤排成一条线。**学会绘制图的人将能够运行一整支 Agent 集群**——而且永远不会感受到其他人受困其中的那层能力天花板。

---

## 原文信息

- 原文：[Codez (@0xCodez) on X](https://x.com/0xCodez/status/2079165300625330317)
- 发布时间：2026 年 7 月 20 日 11:23
- 原始文件：`raw/articles/Graph Engineering with Claude.md`
