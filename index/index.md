---
layout: default
title: Index
---

# Index

## Pages

{% for p in site.pages %}
- `{{ p.path }}` → `{{ p.url }}`
{% endfor %}

## Static files

{% for f in site.static_files %}
- `{{ f.path }}`
{% endfor %}