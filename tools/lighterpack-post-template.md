---
title: 'Wutai Gear List'
date: 2026-06-19 06:00:00 +0800
permalink: /posts/wutai-gear-list/
categories: [Life]
tags: [trail-running, gear, Wutai]
---

这里写这次路线、天气、补给策略，或者为什么这套装备和上一次不一样。

原始页面：[LighterPack - TITLE](https://lighterpack.com/r/PACK_ID)

> 下面这块不走第三方 script。数据先抓到仓库里的 `_data/lighterpack/PACK_ID.yml`，再由本地模板渲染成静态 HTML，所以部署后仍然是纯静态页面，但版式会尽量贴近 LighterPack。装备信息统一维护在 `_data/gear.yml`，各活动通过引用复用。

{% include lighterpack.html id="PACK_ID" show_title="false" %}

---

使用步骤：

1. 先在 LighterPack 建好新清单，拿到分享链接，例如 `https://lighterpack.com/r/abc123`。
2. 执行：`ruby tools/lighterpack_sync.rb https://lighterpack.com/r/abc123`。这会：
   - 自动把新装备追加到 `_data/gear.yml`（已有装备按名称匹配复用）。
   - 生成引用格式的 `_data/lighterpack/abc123.yml`。
3. 复制这份模板到 `_posts/YYYY-MM-DD-your-slug.md`，把标题、日期、permalink、tags 和 `PACK_ID` 替换掉。

装备在不同活动中共享，可以在 [Gear](/gear/) 页面查看每件装备的使用记录。