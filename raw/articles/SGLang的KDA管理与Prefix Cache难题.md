---
title: "SGLang的KDA管理与Prefix Cache难题"
source: "https://www.xiaohongshu.com/explore/6a5f043f0000000006012450?app_platform=ios&app_version=9.38.1&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBfwP6ZG5KpMnxCVvMxxhMTCopdMCu2nEckzshVaQiYYo=&author_share=1&xhsshare=WeixinSession&shareRedId=ODZDRDU1PD82NzUyOTgwNjY0OTc3ST05&apptime=1784620923&share_id=75c4ee8d72f0493086edb3e7630ba632&wechatWid=ba236b566b0fd1bf09d09c00bd95b398&wechatOrigin=menu"
author:
  - "[[zR（探索方向中）]]"
published: 2026-07-25
created: 2026-07-25
description: "3 亿人的生活经验，都在小红书"
tags:
  - "clippings"
---
![](https://sns-webpic-qc.xhscdn.com/202607251428/d108fc2c48052152e1ad5acec0536103/spectrum/1040g34o322sfedv0mukg5o8jgllg8hllj7voroo!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/3282684273f571edf92d523e6a94b3bd/spectrum/1040g34o322sfedv0mue05o8jgllg8hllmdon6j0!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/7cc3fab1db43ab4ae96de56dce415643/spectrum/1040g34o322sfedv0mueg5o8jgllg8hlli9qbom8!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/c34047d1af4c871bbe7c181d74347178/spectrum/1040g34o322sfedv0muf05o8jgllg8hll9kn4l3o!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/c28a6cc548f0083d94ad3e8556652aa4/spectrum/1040g34o322sfedv0mufg5o8jgllg8hllu270dso!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/e6ec02445076f06d22682ef59007bd85/spectrum/1040g34o322sfedv0mug05o8jgllg8hllbppk9eg!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/3d40d24a4388356008bc067273f03847/spectrum/1040g34o322sfedv0mugg5o8jgllg8hllglbmnp0!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/d818879393609b8655a5b38d3cc52d38/spectrum/1040g34o322sfedv0muh05o8jgllg8hllhno1t6g!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/8bf9182b6e345f15fe7baa0422f36623/spectrum/1040g34o322sfedv0muhg5o8jgllg8hll37mp600!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/c8b5dbdcd29077c3fce15d011b09826b/spectrum/1040g34o322sfedv0mui05o8jgllg8hllrk0lho0!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/7a25333bd2c7f30641e7598b8b17aa4c/spectrum/1040g34o322sfedv0muig5o8jgllg8hllnep9370!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/34eda099ebf383657e54755feb90ecce/spectrum/1040g34o322sfedv0muj05o8jgllg8hll4fo7qs8!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/84baf8bb3c0ab490b1e242e51e82846a/spectrum/1040g34o322sfedv0mujg5o8jgllg8hllkbr9e1o!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/7e9732a550d4c061da58fe0c2f89616b/spectrum/1040g34o322sfedv0muk05o8jgllg8hllad86kg8!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/d108fc2c48052152e1ad5acec0536103/spectrum/1040g34o322sfedv0mukg5o8jgllg8hllj7voroo!nd_dft_wlteh_webp_3) ![](https://sns-webpic-qc.xhscdn.com/202607251428/3282684273f571edf92d523e6a94b3bd/spectrum/1040g34o322sfedv0mue05o8jgllg8hllmdon6j0!nd_dft_wlteh_webp_3)

1/14

又是深夜看代码的一天，分享自己的学习路径。 期待开源社区的更新。 [#大模型](https://www.xiaohongshu.com/search_result?keyword=%25E5%25A4%25A7%25E6%25A8%25A1%25E5%259E%258B&type=54&source=web_note_detail_r10) [#开源](https://www.xiaohongshu.com/search_result?keyword=%25E5%25BC%2580%25E6%25BA%2590&type=54&source=web_note_detail_r10) [#K3](https://www.xiaohongshu.com/search_result?keyword=K3&type=54&source=web_note_detail_r10) [#KDA](https://www.xiaohongshu.com/search_result?keyword=KDA&type=54&source=web_note_detail_r10) [#SGLang](https://www.xiaohongshu.com/search_result?keyword=SGLang&type=54&source=web_note_detail_r10) [#kvcache](https://www.xiaohongshu.com/search_result?keyword=kvcache&type=54&source=web_note_detail_r10) [#学习日记](https://www.xiaohongshu.com/search_result?keyword=%25E5%25AD%25A6%25E4%25B9%25A0%25E6%2597%25A5%25E8%25AE%25B0&type=54&source=web_note_detail_r10) [#算法](https://www.xiaohongshu.com/search_result?keyword=%25E7%25AE%2597%25E6%25B3%2595&type=54&source=web_note_detail_r10)

共 13 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31mjn7uq2mm6g5pfn21sipoll38gm15g?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/65f71079000000000b00e2b5?channel_type=web_note_detail_r10&xsec_token=ABdPCDBDj0trwS2I5GBxig349NG2cCBRMZWh0gq8f9vQA%3D&xsec_source=pc_comment)

确实读爽了

2

1

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31f4568kdmq005otu688pgli6um7psl0?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/63be32110000000026005646?channel_type=web_note_detail_r10&xsec_token=ABob3UgnvtTbC-UO6OhdAz92yh2OiugZXIwmLlrzpUFOY%3D&xsec_source=pc_comment)

报名前排听 zR 老师授课

3

1

带带

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31i0dr7k3n80g5piobmigudoisi7k25o?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/66585da50000000003033712?channel_type=web_note_detail_r10&xsec_token=ABIPgrVb-lU30RAS41Mzl6s4JCSQhui8MuBpXHCMxsV9M%3D&xsec_source=pc_comment)

报名前排听 zR 老师授课

3

1

大佬谦虚了

1

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31icp4d0i7s5g5o69n160bksfiuq1vto?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/60c9b84c000000000101d38f?channel_type=web_note_detail_r10&xsec_token=ABnvpi4kjyMP7fMp_2AJmIsQOYhxGCtvAvGJfsOI5wWqA%3D&xsec_source=pc_comment)

什么时候开课zR老师

1

1

带带弟弟

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31r7pfm8hg0505pes6f32u0nhv4d9gh8?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/65dc33c6000000000b0302f1?channel_type=web_note_detail_r10&xsec_token=ABm3AC6Tpn9CCeYpIbaFDsIY6qZQ1v0ke7MCFZcpb_wOY%3D&xsec_source=pc_comment)

靠，这讲的太详细了，读爽了。

2

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/645b6b2504c28aa15eab1dac.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/67471f13000000001c018f5c?channel_type=web_note_detail_r10&xsec_token=ABTa7BK3hFmBvMkQ4JGLKPUyj3Uico3qNTBi7vYtdsAlQ%3D&xsec_source=pc_comment)

看起来和KDA本身性质关系不算大，只要是线性注意力模型都会遇到这种问题，所以理论上不需要等K3开源就已经有很多优化方案了？千问3.5虽然是旧的GDN但是缓存状态是一样的，看起来这篇文章关注点也不是计算相关的算子，所以逻辑一致？

1

3

对，GDN和KDA在SGLang里用同一套MambaPool + checkpoint机制。K3的技术博客提到会给vLLM提供一个优化吧，不知道sglang这会不会有，期待的是这个

赞

回复

展开 2 条回复

登录查看全部评论内容

登录后评论

204263

可以添加到收藏夹啦

13

<iframe src=""></iframe>

鼠标悬停查看 Ta 的信息 好的