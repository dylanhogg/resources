---
layout: default
title: Index
---

# Index

{% assign pages = site.pages | sort: "path" %}

{% for p in pages %}
{% if p.ext == ".md" and p.dir == "/" %}

- [{{ p.name | remove: ".md" }}]({{ p.url | relative_url }})
  {% endif %}
  {% endfor %}
