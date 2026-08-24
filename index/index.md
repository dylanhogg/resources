---
layout: default
title: Index
---

# Index

{% assign pages = site.pages | sort_natural: "path" %}

{% for p in pages %}
  {% assign ext = p.path | slice: -3, 3 %}

  {% if ext == ".md" and p.path != "index/index.md" %}
    {% assign parts = p.path | split: "/" %}
    {% assign depth = parts.size | minus: 1 %}
    {% assign indent = depth | times: 24 %}

    <div style="margin-left: {{ indent }}px">
      - <a href="{{ p.url | relative_url }}">{{ parts | last | remove: ".md" }}</a>
    </div>
  {% endif %}
{% endfor %}