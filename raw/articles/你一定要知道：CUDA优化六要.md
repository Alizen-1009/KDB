---
title: "你一定要知道：CUDA优化六要"
source: "https://www.xiaohongshu.com/explore/69dce8be000000002202586c?xsec_token=ABJbLpTL4P9loytShJqgX8S8GsRngKy4a01a2FPdd7nTw=&xsec_source=pc_collect"
author:
  - "[[简行AI关注]]"
  - "[[简行AI作者]]"
published:
created: 2026-04-22
description: "#大模型 #量化 #编程 #ai #infra 	 cuda优化六要素 	 Global Memory 合并 — 减少事务数 	 Shared Memory bank conflict — 减少串行等待 	 Occupancy — 寄存器/shared mem/线程数的平衡，保证足够 warp 隐藏延迟 	 算法层面减少访存量 — tiling、数据复用（一次搬进 shared mem，多次计算用） 	 指令级 — 避免 warp divergence（同一 warp 内 if/else 两个分支都要跑，串行化） 	 Launch 配置 — Block 数量要远多于 SM 数，避免 tail effect（最后一波 Block 只占用部分 SM）"
tags:
  - "clippings"
---
![](https://sns-webpic-qc.xhscdn.com/202604221110/cfe9c14f778e5d06f2fe0f60332607c3/notes_pre_post/1040g3k031utdghuiia305nir42a092fo1c24j7o!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/33671ffa27c12cbf1b8a4455df47899b/notes_pre_post/1040g3k031utdghuiia005nir42a092fo04ples0!nd_dft_wgth_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/ea3b33ad24e42eb6b6a3722196a655af/notes_pre_post/1040g3k031utdghuiia0g5nir42a092fof4boai8!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/3b2f52a1c0dc8974351f73bbf683fd2d/notes_pre_post/1040g3k031utdghuiia105nir42a092fol4tdvl0!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/834a360d47ea8dc5be5695e938c58deb/notes_pre_post/1040g3k031utdghuiia1g5nir42a092foqij4acg!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/add1e0c3e7b1c12eb6b10c5d2d40a11e/notes_pre_post/1040g3k031utdghuiia205nir42a092fopft0e8g!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/25d66e093bc3615c9fc657e262bc66d2/notes_pre_post/1040g3k031utdghuiia2g5nir42a092fo8b36ev8!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/cfe9c14f778e5d06f2fe0f60332607c3/notes_pre_post/1040g3k031utdghuiia305nir42a092fo1c24j7o!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202604221110/33671ffa27c12cbf1b8a4455df47899b/notes_pre_post/1040g3k031utdghuiia005nir42a092fo04ples0!nd_dft_wgth_webp_3)

2/7

你一定要知道：CUDA优化六要

[#大模型](https://www.xiaohongshu.com/search_result?keyword=%25E5%25A4%25A7%25E6%25A8%25A1%25E5%259E%258B&type=54&source=web_note_detail_r10) [#量化](https://www.xiaohongshu.com/search_result?keyword=%25E9%2587%258F%25E5%258C%2596&type=54&source=web_note_detail_r10) [#编程](https://www.xiaohongshu.com/search_result?keyword=%25E7%25BC%2596%25E7%25A8%258B&type=54&source=web_note_detail_r10) [#ai](https://www.xiaohongshu.com/search_result?keyword=ai&type=54&source=web_note_detail_r10) [#infra](https://www.xiaohongshu.com/search_result?keyword=infra&type=54&source=web_note_detail_r10) cuda优化六要素 Global Memory 合并 — 减少事务数 Shared Memory bank conflict — 减少串行等待 Occupancy — 寄存器/shared mem/线程数的平衡，保证足够 warp 隐藏延迟 算法层面减少访存量 — tiling、数据复用（一次搬进 shared mem，多次计算用） 指令级 — 避免 warp divergence（同一 warp 内 if/else 两个分支都要跑，串行化） Launch 配置 — Block 数量要远多于 SM 数，避免 tail effect（最后一波 Block 只占用部分 SM）

共 1 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31pqmpam2kk005nir42a092fol5tb5v0?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5e5b209400000000010089f8?channel_type=web_note_detail_r10&xsec_token=AButrIVK7EkJ3JzcWGX3pP_QPrWpcF-aBWNegil6JWM0U%3D&xsec_source=pc_comment)

![](http://sns-webpic-qc.xhscdn.com/202604221110/5a82e02a1ff1daae0ad7bdd77c2f00f4/comment/1040g2h031uvbefu42a005nir42a092fo9cj8jlg!nc_n_webp_mw_1)

赞

回复

\- THE END -

说点什么...

24551