---
layout: default
title: Index
---

# Index

{% assign pages = site.pages | sort: "path" %}

{% for p in pages %}
{% if p.ext == ".md" and p.name != "index.md" %}

- [{{ p.path | remove: ".md" }}]({{ p.url | relative_url }})
  {% endif %}
  {% endfor %}
