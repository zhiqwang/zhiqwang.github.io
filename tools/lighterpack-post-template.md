---
title: 'Wutai Gear List'
date: 2026-06-19 06:00:00 +0800
permalink: /posts/wutai-gear-list/
categories: [Life]
tags: [trail-running, gear, Wutai]
---

这里写这次路线、天气、补给策略，或者为什么这套装备和上一次不一样。

原始页面：[LighterPack - TITLE](https://lighterpack.com/r/PACK_ID)

> 下面这块不走第三方 script。数据先抓到仓库里的 `_data/lighterpack/PACK_ID.yml`，再由本地模板渲染成静态 HTML，所以部署后仍然是纯静态页面，但版式会尽量贴近 LighterPack。

{% include lighterpack.html id="PACK_ID" show_title="false" %}

如果后面要同步新版清单，直接执行：`ruby tools/lighterpack_sync.rb https://lighterpack.com/r/PACK_ID`。省略 output 时，会默认写到 `_data/lighterpack/PACK_ID.yml`。

---

使用步骤：

1. 先在 LighterPack 建好新清单，拿到分享链接，例如 `https://lighterpack.com/r/abc123`。
2. 执行：`ruby tools/lighterpack_sync.rb https://lighterpack.com/r/abc123`。这会默认生成 `_data/lighterpack/abc123.yml`。
3. 复制这份模板到 `_posts/YYYY-MM-DD-your-slug.md`，把标题、日期、permalink、tags 和 `PACK_ID` 替换掉。

如果和旧比赛有重复装备，不需要在仓库里做额外去重。LighterPack 那边怎么组织这次清单，这里就按这次快照渲染；不同比赛各自对应一个 YAML 文件。