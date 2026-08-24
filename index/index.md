---
layout: default
title: Index
---

# Index

{% assign pages = site.pages | sort: "path" %}

{% for p in pages %}
  {% assign ext = p.path | slice: -3, 3 %}
  {% if ext == ".md" and p.path != "index/index.md" %}
- [{{ p.path | remove: ".md" }}]({{ p.url | relative_url }})
  {% endif %}
{% endfor %}