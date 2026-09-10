---
title: "Using Claude Code: The unreasonable effectiveness of HTML"
source: "https://claude.com/blog/using-claude-code-the-unreasonable-effectiveness-of-html"
author:
published: 2001-05-20
created: 2026-08-21
description: "How and why members of the Claude Code team use HTML instead of Markdown to produce richer, more readable, and easily shareable outputs."
tags:
  - "clippings"
---
Markdown has become the dominant file format used by agents to communicate with humans. It’s simple, portable, has some rich text capability and is easy to edit. Claude has even gotten surprisingly good at using ASCII to make diagrams inside of Markdown files.  
Markdown 已成为智能体与人类沟通的主要文件格式。它简单易用、便于携带、具备一定的富文本功能，而且易于编辑。克劳德甚至已经非常擅长使用 ASCII 码在 Markdown 文件中绘制图表。

But as agents have become more and more powerful, I’ve found that Markdown has become an increasingly restrictive format. Specifically, I find it difficult to read a Markdown file of more than a hundred lines; I want to use Claude to generate richer visualizations, color and diagrams; and I want to be able to share these outputs more easily.

I also am increasingly not editing these files myself, but using them as specs and reference files. When I do make edits, I’m usually prompting Claude to edit them, which removes one of Markdown’s largest benefits.

Instead, I’ve started preferring HTML as an output format instead of Markdown and increasingly see this pattern being applied by others on the Claude Code team. In this post, I share why and how our team uses HTML to produce richer, more readable Claude Code outputs. If you'd like to follow along, you can start using these [HTML file templates](https://thariqs.github.io/html-effectiveness/#code-review) for common use cases, too.

## Why use HTML?

A few things make HTML a better fit than Markdown for the kind of work I'm now doing with Claude Code, including tasks that require or entail:

## Information density

![__wf_reserved_inherit](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cc2df7520821249c2495c_image10.png)

\_\_wf\_reserved\_inherit

HTML can convey much richer information compared to Markdown. It can, of course, do simple document structure like headers and formatting, but it can also represent all sorts of other information such as:

- Tabular data using tables
- Design data with CSS
- Illustrations with SVG
- Code snippets with script tags
- Interactions using HTML elements with javascript + CSS  
	使用 HTML 元素结合 JavaScript 和 CSS 进行交互
- Workflows using SVG and HTML
- Spatial data using absolute positions and canvases
- Images using image tags

In my opinion, there is almost no set of information that Claude can read that you cannot efficiently represent with HTML. This makes it a highly efficient way for the model to communicate in-depth information to you and for you to review it.

I’ve found that in the absence of being able to do this, the model may do more inefficient things in Markdown, like ASCII diagrams or, my favorite, estimating colors with unicode characters.

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb4e_be6aa05f.png)

## Visual clarity and ease of reading

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb48_343de6c4.png)

As Claude is capable of tackling more complex work, it's also able to write larger and larger specs and plans. I’ve found that I tend to not actually read more than a 100-line Markdown file, and I certainly am not able to get anyone else in my organization to read it.

But HTML documents are much easier to read because Claude can organize the structure visually to be ideal to navigate with tabs, illustrations, and links. It can even be mobile responsive so you can read it differently based on your form factor.

## Ease of sharing

Markdown files are fairly hard to share since most browsers do not render them natively well. You often have to add them as attachments to emails or messages.

As long as you upload the HTML file, you can share the link easily. Your colleagues can open it wherever they wish and easily reference it.

The chance of someone actually reading your spec, report, or PR writeup is much higher if it’s in HTML.

## Two-way interactions

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb4b_438fa236.png)

HTML can also allow you to [interact with the document](https://x.com/trq212/status/2017024445244924382); for example, you might want to ask it to add sliders or knobs to adjust a design or allow you to tweak different options in the algorithm to see what happens. You can also ask it to let you copy these changes into a prompt to paste back into Claude Code.

When useful, this can allow you to create individual editing environments for the specific problem you’re working on.

## Data ingestion

One of the biggest reasons to use Claude Code to make HTML files instead of Claude.ai or Claude Design is all of the context Claude Code can ingest. For example, when writing this article, I asked Claude Code to read through my code folder and find all the HTML files I've generated, group and categorize them, and then make an HTML file with diagrams representing each type. The diagrams you see in this article are a direct result of that.

Besides the file system, Claude Code can find additional context using your MCPs (like Slack, Linear, etc.), your web browser (with Claude in Chrome), and your git history.

## Getting started

One thing worth noting: you don't need to do much to get Claude to generate HTML like this. You can simply prompt it to " *make an HTML file* " or " *make an HTML artifact*." The main thing is knowing what you want the artifact to do and how you might use it. Over time, it may make sense to build a skill around recurring patterns, but starting by prompting from scratch is a good way to get a feel for how it works across different use cases.

## Use cases

To make this approach more concrete, below are some [example use cases](https://thariqs.github.io/html-effectiveness/) where I think using HTML files make more sense than Markdown. You can also follow along with a GitHub gallery of these use cases, [here](https://github.com/anthropics/html-effectiveness).

### Specs, planning, and exploration

HTML is a rich canvas for Claude to dive into a problem. When I start working on a problem instead of a simple Markdown plan I expect to make a web of HTML files. For example, I might start with asking Claude Code to brainstorm and create some explorations of different options. I would then ask it to expand more into one, maybe make mockups or examples of the type interfaces. Finally, when I feel good I’ll ask it to write an implementation plan. When I’m happy with the plan I’ll create a new session and pass in all of these files for it to implement.

When verifying I’ll also ask the verification agent to read in the files and it will have much broader context on what is needed.

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb5c_7eaa0090.png)

**Example prompts:**

- *I'm not sure what direction to take the onboarding screen. Generate 6 distinctly different approaches—vary layout, tone, and density—and lay them out as a single HTML file in a grid so I can compare them side by side. Label each with the tradeoff it's making.*
- *Create a thorough implementation plan in a HTML file, be sure to make some mockups, show data flow and add important code snippets I might want to review. Make it easy to read and digest.*

**Use this for:**

- Exploring other ways to implement something in code
- Experimenting with multiple visual designs at once  
	同时尝试多种视觉设计

### Code review and understanding 代码审查和理解

Code can be difficult to read in a Markdown file, but with HTML, we can render diffs, annotations, flowcharts, and modules. Use HTML to understand code that the agent has written, to review code, or to explain a PR to someone reviewing your code.  
Markdown 文件中的代码可能难以阅读，但使用 HTML，我们可以渲染差异、注释、流程图和模块。使用 HTML 可以更好地理解代理编写的代码、审查代码，或者向审查代码的人解释 PR（Pull Request，拉取请求）。

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb5f_ce1ada20.png)

**Example prompt: 示例提示：**

*Help me review this PR by creating an HTML artifact that describes it. I'm not very familiar with the streaming/backpressure logic, so focus on that. Render the actual diff with inline margin annotations, color-code findings by severity and whatever else might be needed to convey the concept well.  
请帮我审核这个 PR，创建一个 HTML 文件来描述它。我对流式传输/反压逻辑不太熟悉，所以请重点关注这部分。请渲染实际的差异，添加内联边距注释，根据严重程度对发现的问题进行颜色编码，以及其他任何有助于清晰表达概念的内容。*

**Use this for: 用途：**

- Creating a PR 创建公关稿
- Reviewing a PR 审核公关稿
- Understanding a topic in code  
	理解代码中的某个主题

### Design and prototypes 设计和原型

Claude Design is based on HTML because HTMLis incredibly expressive at design, even if your end surface is not HTML. Claude can sketch out a design in HTML and then write it in your language of choice, be it React, Swift, etc.  
Claude Design 基于 HTML，因为 HTML 在设计方面极具表现力，即使你的最终界面并非 HTML。Claude 可以用 HTML 勾勒出设计草图，然后用你选择的语言（例如 React、Swift 等）将其编写出来。

You can also prototype interactions, such as animations, actions, etc. Consider asking Claude to make sliders, knobs, etc. to tune in exactly what you’re looking for.  
您还可以制作交互原型，例如动画、动作等。不妨请 Claude 制作滑块、旋钮等，以便精确调整到您想要的效果。

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb51_2f351343.png)

**Example prompt: 示例提示：**

I want to prototype a new checkout button, when clicked it does a play animation and then turns purple quickly. Create a HTML file with several sliders and options for me to try different options on this animation, give me a copy button to copy the parameters that worked well.  
我想制作一个新的结账按钮原型，点击后会播放一段动画，然后迅速变成紫色。请创建一个包含多个滑块和选项的 HTML 文件，以便我尝试不同的动画效果，并提供一个复制按钮，方便我复制效果好的参数。

**Use this for:用途：**

- Creating design system artifacts  
	创建设计系统工件
- Adjusting components 调整部件
- Visualizing component libraries  
	可视化组件库
- Prototyping animations 动画原型

### Reports, research, and learning报告、研究和学习

Claude Code is very effective at synthesizing information across multiple data sources and converting it into a report for readability. You can prompt Claude to search your Slack, your codebase, git history, or the internet and use it to generate easy to read reports..  
Claude Code 能够高效地整合来自多个数据源的信息，并将其转换为易于阅读的报告。您可以让 Claude 搜索您的 Slack、代码库、Git 历史记录或互联网，并利用这些信息生成易于阅读的报告。

You could assemble this in the form of a long HTML document, an interactive explainer or even a slideshow/deck. Ask Claude to use SVG for diagrams to help visualize it.  
你可以把它整理成一个长篇 HTML 文档、一个交互式讲解器，甚至是幻灯片/演示文稿。让克劳德用 SVG 格式绘制图表，以便更好地呈现。

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb54_e0306f54.png)

**Example prompt:示例提示：**

*I don't understand how our rate limiter actually works. Read the relevant code and produce a single HTML explainer page: a diagram of the token-bucket flow, the 3–4 key code snippets annotated, and a "gotchas" section at the bottom. Optimize it for someone reading it once.  
我不明白我们的限速器是如何工作的。请阅读相关代码，并制作一个 HTML 解释页面：包含令牌桶流程图、3-4 个关键代码片段的注释，以及底部的“注意事项”部分。请针对只阅读一次的用户进行优化。*

**Use this for:用途：**

- Writing feature summarizations  
	撰写专题摘要
- Generating explainers 生成解释器
- Drafting weekly status reports  
	撰写每周状态报告
- Creating incident reports  
	创建事件报告
- Producing SVG illustrations, flowcharts, and technical diagrams,  
	制作 SVG 插图、流程图和技术图表，

### Custom editing interfaces自定义编辑界面

Sometimes it’s hard to describe what you want purely in a text box. For this use case, I'll often ask Claude to build me a throwaway editor for the exact thing I'm working on: not a product, or a reusable tool, but a single HTML file, purpose-built for this one piece of data.  
有时候，仅仅用文本框很难准确描述你想要的东西。在这种情况下，我通常会请 Claude 为我正在处理的特定数据创建一个临时编辑器：不是一个产品，也不是一个可复用的工具，而是一个专门为这一条数据而构建的 HTML 文件。

The trick is always to end with an export: a "copy as JSON" or "copy as prompt" button that turns whatever I did in the UI back into something I can paste into Claude Code or commit to a file. You stay in the loop, but the loop gets much tighter.  
诀窍在于始终以导出操作结束：添加一个“复制为 JSON”或“复制为提示符”按钮，将我在用户界面中所做的任何操作转换回可以粘贴到 Claude Code 或提交到文件的格式。这样你仍然处于循环中，但循环变得更加紧密。

![](https://cdn.prod.website-files.com/68a44d4040f98a4adf2207b6/6a0cccd2c085977fc720eb57_0e3ace42.png)

**Example prompts:示例提示：**

- *I need to reprioritize these 30 Linear tickets. Make me an HTML file with each ticket as a draggable card across Now / Next / Later / Cut columns. Pre-sort them by your best guess. Add a "copy as Markdown" button that exports the final ordering with a one-line rationale per bucket.  
	我需要重新调整这 30 个 Linear 任务的优先级。请创建一个 HTML 文件，其中每个任务都以卡片的形式呈现，可以在“立即处理”、“下一步处理”、“稍后处理”和“取消处理”这几列之间拖动。请根据您的最佳判断预先排序。添加一个“复制为 Markdown”按钮，导出最终排序结果，并为每个类别添加一行说明。*
- *Here's our feature flag config. Build a form-based editor for it, group flags by area, show dependencies between them, warn me if I enable a flag whose prerequisite is off. Add a "copy diff" button that gives me just the changed keys.  
	这是我们的功能标志配置。请为其构建一个基于表单的编辑器，按领域对标志进行分组，显示它们之间的依赖关系，并在我启用某个标志但其先决条件未满足时发出警告。添加一个“复制差异”按钮，仅提供已更改的键值。*
- *I'm tuning this system prompt. Make a side-by-side editor: editable prompt on the left with the variable slots highlighted, three sample inputs on the right that re-render the filled template live. Add a character/token counter and a copy button.  
	我正在调整这个系统提示。创建一个并排编辑器：左侧是可编辑的提示框，变量槽位高亮显示；右侧是三个示例输入框，可以实时重新渲染已填充的模板。添加字符/标记计数器和复制按钮。*

**Use this for:用途：**

- Reordering, triaging, or bucketing anything (tickets, test cases, feedback)  
	对任何内容（工单、测试用例、反馈）进行重新排序、分类或归类
- Editing structured config (feature flags, env vars, JSON/YAML with constraints)  
	编辑结构化配置（功能标志、环境变量、带有约束的 JSON/YAML）
- Tuning prompts, templates, or copy with live preview  
	带有实时预览的调校提示、模板或文案
- Curating datasets — approve/reject rows, tag examples, export the selection  
	管理数据集——批准/拒绝行、标记示例、导出选定内容
- Annotating a document, transcript, or diff and exporting the annotations  
	对文档、转录稿或差异进行注释并导出注释
- Picking values that are painful to express in text: colors, easing curves, crop regions, cron schedules, regexes  
	选择那些难以用文字表达的值：颜色、缓动曲线、裁剪区域、定时任务、正则表达式

### Frequently asked questions常见问题解答

These are the questions I get asked most often about using HTML with Claude Code, paired with the practical, day-to-day habits I've landed on:  
以下是我在使用 Claude Code 编写 HTML 代码时最常被问到的问题，以及我总结出的实用日常习惯：

**Isn’t it less efficient?效率不是更低吗？**

While Markdown often uses fewer tokens, I’ve found that the added expressiveness of HTML and the much higher likelihood of me reading it means I get overall better output. With the 1MM context window in Opus 4.7, the increased token usage is not really noticeable in the context window.  
虽然 Markdown 通常使用的标记较少，但我发现 HTML 更丰富的表达能力以及我阅读它的可能性更高，这意味着我能获得更好的整体输出效果。在 Opus 4.7 的 100 万字上下文窗口中，标记使用量的增加在上下文窗口中并不明显。

**When do you use Markdown for now?  
目前你什么时候使用 Markdown？**

I have honestly stopped using Markdown altogether for almost everything, but I’m probably far on the HTML maximalist side of things.  
说实话，我已经完全停止在几乎所有地方使用 Markdown 了，但我可能在 HTML 的使用上属于极端主义者。

**Is this how you’ve replaced planning?  
这就是你们取代计划的方式吗？**

I’ve found that instead of having a single plan, I tend to have a few different HTML files for different parts/stages of the plan. For example, I may make an implementation plan in HTML and then do another file for exploration of UIs, and then finally make a HTML component that lists every design. I tend to keep these files around as references for the future, as well for use in verification.  
我发现，与其只制定一个计划，我更倾向于为计划的不同部分/阶段创建几个不同的 HTML 文件。例如，我可能会用 HTML 创建一个实现计划，然后用另一个文件来探索用户界面，最后创建一个 HTML 组件来列出所有设计方案。我通常会保留这些文件，以便将来参考以及用于验证。

## Staying in the loop with Claude与克劳德保持联系

All of the above is to say that the real reason I use HTML instead of Markdown is that it helps me feel much more in the loop with Claude. As Claude takes on more, I'd noticed I was reading plans less closely, and I wanted a way to stay engaged with its choices rather than just hand them off. HTML turned out to be exactly that. I feel more in the loop now than I ever did before."  
以上种种都表明，我使用 HTML 而不是 Markdown 的真正原因是，它让我感觉与 Claude 的沟通更加顺畅。随着 Claude 承担的工作越来越多，我发现自己阅读计划的细致程度有所下降，因此我需要一种方法来参与到他的决策过程中，而不是仅仅把决策权交给他。HTML 正好满足了我的需求。现在，我感觉自己比以往任何时候都更了解整个流程。

Get started with [Claude Code](https://claude.com/product/claude-code).  
开始使用 [Claude Code](https://claude.com/product/claude-code) 。

*This article was written by Thariq Shihipar, member of technical staff, and expresses his personal opinions – and affinity – for using HTML files with Claude Code*.  
*本文由技术人员 Thariq Shihipar 撰写，表达了他对使用 Claude Code 处理 HTML 文件的个人观点和偏好* 。

## Transform how your organization operates with Claude借助 Claude 改变您组织的运营方式

Product updates, how-tos, community spotlights, and more. Delivered monthly to your inbox.  
产品更新、使用指南、社区亮点等精彩内容，每月发送至您的邮箱。