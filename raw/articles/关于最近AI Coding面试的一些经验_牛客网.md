---
title: "关于最近AI Coding面试的一些经验_牛客网"
source: "https://www.nowcoder.com/discuss/914899592678805504?sourceSSR=post"
author:
published:
created: 2026-09-05
description: "经历了n次面试和拷打之后，明显感觉到今年暑期实习面试已经跟以前不一样了随着Agent越来越热，现在的面试，尤其是一面，越来越喜欢考现场AI Coding这也意味着不能再按传统节奏埋头刷LeetCode了，得换个思路准备结合近期面了几家大厂的经验，整理了一些内容供大家参考，尽量帮各位避开一些坑一、大厂_牛客网_牛客在手,offer不愁"
tags:
  - "clippings"
---
[![头像](https://uploadfiles.nowcoder.com/images/20260731/116086861_1785466906055/E2BB4E6C0666DF6F99D6A2A58463BC37?x-oss-process=image%2Fresize%2Cw_72%2Ch_72%2Cm_mfit)](https://www.nowcoder.com/users/116086861)

08-06 15:24 门头沟学院 全栈开发 发布于广东

经历了n次面试和拷打之后，明显感觉到今年暑期实习面试已经跟以前不一样了

随着Agent越来越热，现在的面试，尤其是一面，越来越喜欢考现场AI Coding

这也意味着不能再按传统节奏埋头刷LeetCode了，得换个思路准备

结合近期面了几家大厂的经验，整理了一些内容供大家参考，尽量帮各位避开一些坑

**一、大厂为什么开始推AI Coding面试**

之前跟几个HR聊过，感觉这不是临时调整，后面大概率会有越来越多公司引入AI Coding笔试/面试。现在工作中谁还自己吭哧吭哧写完所有代码啊，都是指挥AI帮你干活，自己把控方向和质量就行

所以招聘的时候当然要看看候选人会不会跟AI配合干活

另外还有一个背景——传统笔试里不少候选人已经开始用AI作弊了，所以企业索性把AI直接纳入考核环境，考察的是候选人能不能合理使用这类工具

不具备这个能力的，自然会在这环节被筛掉

**二、AI Coding面试和AI Coding笔试是两回事**

很多同学以为AI Coding面试只是把笔试的形式搬到面试里，核心还是靠AI写完代码就行

我觉得这两者差别还挺大的

AI Coding笔试是考试者独立跟AI协作完成，全程没有面试官介入，核心考察的是借助AI产出可运行代码的能力，由机器自动评分，只看最终代码能不能通过测试用例，过程数据基本不作评分依据

这就意味着就算你提示词写得一团糟，只要AI碰巧给你全改对了，提交上去依然能过

但AI Coding面试不一样——面试官会全程跟进，考察你指挥AI完成开发、写提示词、管理上下文、组织测试和架构的全流程能力，主要靠人工评估，更看重你对需求的拆解思路、Prompt设计迭代的逻辑、以及定位和修复问题的能力

而且据我了解，在AI Coding面试中，你跟AI的对话记录、Prompt调整次数、Token消耗情况都会被完整记录，面试官随时可以调出来看

另外AI Coding面试一般是30-40分钟，笔试通常是2小时，所以面试题一般会比笔试简单一些

大概区别就这些

**三、AI Coding面试到底在考什么**

大部分面试官明确说过他们并不太关注最终代码能不能跑通，主要看以下几项能力（我自己总结的）：

**1\. 需求拆解能力**

能不能把复杂需求拆成AI能理解、能执行的小任务，而不是直接扔一句模糊指令

AI Coding面试的题目通常不会很长，就两三句话，需求比较简略

这个时候一定要让AI先做需求分析，把需求拆开。注意不要直接复制原题，最好用自己的话扩充一下再喂给AI

**2\. 提示词边界控制能力**

能不能给AI明确输出约束——比如指定技术栈和版本、明确禁止用哪些库、要求覆盖哪些边界场景

如果你对这个题目本身不太熟，可以先让AI推荐技术栈、描述边界情况，然后你自己做取舍就行

**3\. 校验迭代能力**

能不能识别AI输出中的错误，并通过调Prompt来优化。有些公司的模型确实不太好用，容易产生幻觉、遗漏边界场景或漏掉需求。如果你识别不了这些问题、直接全盘接受AI的输出，可能会被打上不好的标签

建议在AI生成的同时自己扫一眼代码和文档，别干等。AI写完之后可以另开一个对话，让一个全新的AI根据需求文档判断当前代码是否已经完整实现了需求，如果没实现再丢回给第一个AI继续改

**四、我自己常用的提示词模板**

AI Coding面试里Prompt质量直接影响评估结果

一套完整的提示词大概需要以下几种：需求分析、技术方案、代码实现、测试迭代、总结陈述

**1\. 需求分析提示词**

模板：请作为资深后端开发工程师，对以下编程需求进行全面分析

请明确核心业务逻辑、关键功能模块、需要处理的边界条件、潜在技术难点，以及输入输出规范，无需编写代码，仅输出结构化的需求分析报告

**2\. 技术方案设计提示词**

模板：基于上述需求分析结果，设计一套可落地的技术实现方案

需明确使用的编程语言、数据结构、核心算法思路、模块划分逻辑、执行流程，确保方案符合工业级开发规范，兼顾代码可读性与运行效率

**3\. 代码实现提示词**

模板：请严格按照上述技术方案，编写完整可运行的代码

需遵循【指定编程语言】编码规范，添加清晰注释，覆盖所有边界场景与异常处理逻辑，禁止省略核心实现逻辑，代码需直接通过编译并满足需求要求

技术约束：【补充语言版本、禁用库、性能要求等】

**4\. 测试与迭代优化提示词**

模板：请对上述代码进行全面测试分析，列出常规测试用例、边界测试用例、异常测试用例，并校验代码是否存在逻辑漏洞、性能瓶颈、语法错误

针对存在的问题，给出优化后的代码版本，说明优化思路与改进点

**5\. 面试总结陈述提示词**

模板：请对本次编程任务的实现过程进行总结，梳理需求核心要点、技术方案选型原因、代码实现关键逻辑、测试优化内容，以及整体开发思路，形成简洁规范的总结陈述，写成一份README.md

```java
## Role
...
## Task
...
## Context
...
## Constrain
...
```

另外还有几个小技巧：

1. 指令写具体一点，别太模糊
2. 分模块推进，别一次性生成全部代码（特别简单的可以一次写完）
3. 先让AI确认思路，再生成代码
4. 复杂问题加一句“请逐步思考”

**五、几个容易踩的坑**

**1\. 过度依赖AI，自己不动脑子**

全程让AI自主完成任务，面试官一问思路就答不上来。正确做法是在AI干活的时候翻一翻它写的PRD和技术文档，大概知道实现方式就行

**2\. 不校验AI输出，直接拿来用**

大模型的幻觉是普遍存在的，不校验就使用会让面试官觉得你缺乏问题识别能力。最好让AI自己审一遍自己，确认没问题再继续

**3\. 反复无效调整，浪费时间和Token**

主要发生在测试用例没跑通的时候，因为无效的提示词导致AI做了一堆无用修改。这块我暂时没遇到特别严重的，大家可以搜一下别人的经验

![](https://uploadfiles.nowcoder.com/images/20260806/116086861_1786000794632/F4CBA416A7AFF1D9D19AECDFD9217B39) ![](https://uploadfiles.nowcoder.com/images/20260806/116086861_1786000781096/D2B5CA33BD970F64A6301FA75AE2EB22) #牛客AI配图神器#

[#牛友的AICoding日常#](https://www.nowcoder.com/creation/subject/f2214037a569458b9c2baf57fe2140ef)

4 11 80 ![](https://static.nowcoder.com/fe/file/oss/1719298000662ECICZ.png)

浏览 1737

大家都在搜：笔试作弊

一键发评

求面经

哪家在面

蹲个链接

快捷表情

畅所欲言吧～

图片

话题

09-01 21:25

[门头沟学院 C++](https://www.nowcoder.com/users/682271768)

[深信服 软件开发(C++) 一面](https://www.nowcoder.com/discuss/924412590531379200?sourceSSR=post)[1\. 请做一下自我介绍2. ELF 动态链接过程中，PLT、GOT 和延迟绑定是如何配合工作的？位置无关代码不能直接写死外部函数的运行时地址，因此会通过 PLT 跳板和 GOT 表项进行间接调用。程序第一次调用外部函数时... 查看更多](https://www.nowcoder.com/discuss/924412590531379200?sourceSSR=post)[C++ 常考面试题总结](https://www.nowcoder.com/creation/manager/columnDetail/MJ4oG8)

[Zechariah](https://www.nowcoder.com/users/883382140)

08-25 19:50

[MiniMax\_大模型算法工程师(实习员工)](https://www.nowcoder.com/users/883382140)

[AI Coding体验极差](https://www.nowcoder.com/feed/main/detail/b7605bc82f0149309464570db551c43c?sourceSSR=post)[和编程题一样浮于表面，如果说编程题的算法和well defined problem在现实工程开发中很少碰到，那AI Coding的模型和Harness更是八百辈子不会在生产中用一回，在功能极其简陋的Harness上针对能力奇差的模型磨炼... 查看更多](https://www.nowcoder.com/feed/main/detail/b7605bc82f0149309464570db551c43c?sourceSSR=post)[AICoding笔试感受](https://www.nowcoder.com/creation/subject/9d7c81d9d8774ea4a74d2298c9ba3515?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

[叫我看看嘛](https://www.nowcoder.com/users/289358583)

08-31 20:42

[算法工程师](https://www.nowcoder.com/users/289358583)

[从Demo到落地：AI岗必考的Harness、评测与成本控制，你掌握了吗？](https://www.nowcoder.com/discuss/924039236666261504?sourceSSR=post)[AI岗位的面试风向正在发生一次静悄悄的变化。以前只要能聊清楚模型原理、手撕几道算法题，基本就能稳稳过关；现在越来越多的面试官开始把注意力转向工程化能力，尤其是Harness设计、评测体系搭建和成本控制... 查看更多](https://www.nowcoder.com/discuss/924039236666261504?sourceSSR=post)查看9道真题和解析

[Happy\_py](https://www.nowcoder.com/users/377402781)

08-09 23:15

已编辑

[北京房地产职工大学 Java](https://www.nowcoder.com/users/377402781)

[牛客网的aicoding练习题该如何练？](https://www.nowcoder.com/feed/main/detail/5c6f51a72c9243b0a51a4d510333864b?sourceSSR=post)[最近在和群友们探讨代码工具时，注意到牛客自己上线了专门的 AI Coding 练习题库。起初，大家以为这无非是换了个刷题界面，但随着使用的深入，我发现它更像是一个多功能的“技术练兵场”。无论是应对机试、还是... 查看更多](https://www.nowcoder.com/feed/main/detail/5c6f51a72c9243b0a51a4d510333864b?sourceSSR=post)[牛友的AICoding日...](https://www.nowcoder.com/creation/subject/f2214037a569458b9c2baf57fe2140ef?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

08-25 14:29

[米哈游\_前端开发工程师](https://www.nowcoder.com/users/6010920)

[秋招是一场马拉松，不是百米冲刺 🏃♂️ 聊聊心态管理这件事](https://www.nowcoder.com/discuss/921771031386128384?sourceSSR=post)[秋招进行到现在，可能有一些同学开始出现以下症状：面试挂了之后一整天什么都不想做投了没回复就怀疑自己看到别人拿offer就焦虑开始怀疑"我是不是不行"如果中了两条以上，请继续往下看。我看到过优秀的同学，... 查看更多](https://www.nowcoder.com/discuss/921771031386128384?sourceSSR=post)[为了秋招你都做了哪些准备...](https://www.nowcoder.com/creation/subject/7f8971d2aaa6452993b5786d16ac1b4e?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

4

11

80

真题解析

![](https://static.nowcoder.com/fe/file/oss/1719455754858CVQLW.png) ![](https://static.nowcoder.com/fe/file/site/www-web/prod/1.0.495/imageAssets/fb0f8426d41a5025be30.png)

## 全站热榜

- [
	帆软二面 面经 帆软二面 面经
	2.1W
	](https://www.nowcoder.com/feed/main/detail/b22f8f811465486aa912b80efe8f5cd1)
- [
	面试官视角，手把手带大家改一份 Agent项目简历 面试官视角，手把手带大家改一份 Agent项目简历
	8693
	](https://www.nowcoder.com/discuss/924971145856569344)
- [
	美团笔试8887做题技巧 美团笔试8887做题技巧
	6634
	](https://www.nowcoder.com/discuss/925053622944026624)
- [
	9.4 bigo二面面经 9.4 bigo二面面经
	4628
	](https://www.nowcoder.com/feed/main/detail/698a8216d7e8410eb158fbd1fe8dbd6f)
- [
	BIGO-后台开发 一面面经 BIGO-后台开发 一面面经
	4254
	](https://www.nowcoder.com/feed/main/detail/73eca5e384d44dd2a14ebaaff8cd869f)
- [
	秋招AICoding笔试你就这么做！ 秋招AICoding笔试你就这么做！
	4237
	](https://www.nowcoder.com/discuss/924726077291851776)
- [
	小厂实习如梦 小厂实习如梦
	4082
	](https://www.nowcoder.com/feed/main/detail/ce8b065bdb424879ae900744db51ab1a)
- [
	百度前端一面&二面 百度前端一面&二面
	4015
	](https://www.nowcoder.com/feed/main/detail/6211d5f22c8d4904ae220fb82aa29b76)
- [
	关于秋招的求职一点想法 关于秋招的求职一点想法
	3652
	](https://www.nowcoder.com/feed/main/detail/775cc197414548538b174413476b5a2e)
- [
	字节跳动9.3 Agent开发一面面经 字节跳动9.3 Agent开发一面面经
	3477
	](https://www.nowcoder.com/discuss/925342611194286080)

![](https://static.nowcoder.com/fe/file/oss/2025010217358133565033858.png)

## 创作者周榜

![](https://uploadfiles.nowcoder.com/images/20260126/942981681_1769410092733/7401A839FA2D009CB71595588ED11E87)

炒肉多 软件开发top1

京东 后端开发(实习)

10W+

![](https://uploadfiles.nowcoder.com/images/20260509/459616638_1778295702996/E2BB4E6C0666DF6F99D6A2A58463BC37)

upc

百度 全栈开发(实习)

10W+

![](https://uploadfiles.nowcoder.com/images/20250917/49609549_1758091840544/FECD76F09C4EFFA7102ECDBC1795FB3B)

insisting\_ 更新了爆文

合肥工业大学 Java

10W+

![](https://uploadfiles.nowcoder.com/images/20260731/116086861_1785466906055/E2BB4E6C0666DF6F99D6A2A58463BC37)

棒男孩 更新了爆文

门头沟学院 全栈开发

10W+

牢大我想你了T\_T 更新了爆文

广东工业大学 前端工程师

9.8W

![](https://uploadfiles.nowcoder.com/images/20200722/646661816_1595425646546_34F0D4F2D29608136C02E6A37A1F168C)

林小白zii 收藏top1

香港大学 人工智能

9.1W

![](https://uploadfiles.nowcoder.com/images/20251122/664521299_1763776005622/F68893B1786D09C58913D0B179FDC46F)

程序员花海 更新了爆文

复旦大学 Java

8.0W

![](https://static.nowcoder.com/head/header0006.png)

Akhasi 更新了爆文

西安电子科技大学 Java

6.2W

![](https://static.nowcoder.com/head/header0001.png)

有礼貌的芹菜希望年薪百万

东南大学 Java

6.2W

![](https://uploadfiles.nowcoder.com/images/20250211/355903597_1739274351473/FECD76F09C4EFFA7102ECDBC1795FB3B)

码客明 更新了爆文

美团 测试开发

5.8W

正在热议[\# #](https://www.nowcoder.com/creation/subject/81cfda8b162f4b5abbf29eeae76140c8?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)

[

21449次浏览 579人参与

](https://www.nowcoder.com/creation/subject/81cfda8b162f4b5abbf29eeae76140c8?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

34346次浏览 764人参与

](https://www.nowcoder.com/creation/subject/70c2fd0b995a40d2ac3b68f522079428?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

18711次浏览 192人参与

](https://www.nowcoder.com/creation/subject/aa99cc2283b1424db067e3980fdb866f?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

68269次浏览 628人参与

](https://www.nowcoder.com/creation/subject/bff6bef8a4d648168a8ac3b197f94ae5?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

169246次浏览 1139人参与

](https://www.nowcoder.com/creation/subject/a0dd49089a404b13894558989c2cf19a?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

564441次浏览 9742人参与

](https://www.nowcoder.com/creation/subject/14710425d5b74593b2ef7103d293606f?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

27369次浏览 234人参与

](https://www.nowcoder.com/creation/subject/a0c560e49d8a43cb89017f358b7886b1?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

409348次浏览 1919人参与

](https://www.nowcoder.com/creation/subject/d4aa0484bacb402388a95977de79aa1b?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

11490次浏览 38人参与

](https://www.nowcoder.com/creation/subject/eaa453fc4cbc4425bb3ec0601a6bdef0?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

189688次浏览 1090人参与

](https://www.nowcoder.com/creation/subject/303538254fec4c9c9eefd8c44e136ef8?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

130340次浏览 1241人参与

](https://www.nowcoder.com/creation/subject/1c95320d158647acad966a669ee19b8e?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

409257次浏览 3707人参与

](https://www.nowcoder.com/creation/subject/82685ff8a33b46f389ce925ba2942cab?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

111320次浏览 789人参与

](https://www.nowcoder.com/creation/subject/846ec3662c43453dbc3c6a30c3d2ff8c?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

88184次浏览 666人参与

](https://www.nowcoder.com/creation/subject/0fd2a26b3a074126a8c7317ba7a1a70c?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

189487次浏览 1233人参与

](https://www.nowcoder.com/creation/subject/12ce0adc5acb4b63a9e668f833248422?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

92780次浏览 605人参与

](https://www.nowcoder.com/creation/subject/a9bec1a602fd47cf8437bfa48fcf6bc3?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

111763次浏览 807人参与

](https://www.nowcoder.com/creation/subject/e3dfa7b81e5043bda88e86f3850d8082?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

206463次浏览 1327人参与

](https://www.nowcoder.com/creation/subject/71397faa883a491bab489e26ced0bec4?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

69354次浏览 390人参与

](https://www.nowcoder.com/creation/subject/3cfa5d57945b431ba7c4a954bfffd378?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

255285次浏览 2274人参与

](https://www.nowcoder.com/creation/subject/590e2add320b41f189606d921b688ed0?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

240968次浏览 969人参与

](https://www.nowcoder.com/creation/subject/866930bf6f214c4f8bb4cbbdeb4c8bf5?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)[

\# #

39281次浏览 207人参与

](https://www.nowcoder.com/creation/subject/f6906a635cb842ecb46feefa9e0f427c?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)