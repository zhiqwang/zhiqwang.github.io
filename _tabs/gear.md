---
layout: page
icon: fas fa-person-hiking
order: 6
title: Gear
---

装备库：所有装备在这里只定义一次，不同活动通过引用来复用。

{% for entry in site.data.gear %}
  {% assign slug = entry[0] %}
  {% assign gear = entry[1] %}

### {{ gear.name }}

{% if gear.brand %} **品牌** {{ gear.brand }} · {% endif %}**重量** {% if gear.weight_mg >= 1000000 %}{{ gear.weight_mg | divided_by: 1000000.0 }} kg{% else %}{{ gear.weight_mg | divided_by: 1000 }} g{% endif %} {% if gear.description != '' and gear.description %} · {{ gear.description }}{% endif %}

**使用记录：**

{% include gear_usage.html slug=slug %}

---

{% endfor %}
