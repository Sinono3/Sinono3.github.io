---
layout: layouts/layout.html
---

# Aldo Acevedo

Third-year CS student at Universidad Politécnica Taiwán Paraguay, on two-year exchange at National Taiwan University of Science and Technology (2025-2026).

I'm interested in AI safety, and **I'm actively steering my career towards technical alignment research.**
I'm determined to deeply understand the inner workings of ML systems,
and find ways to make them robustly learn human preferences.

In the past, I've worked as software engineer, primarily in Rust, although not exclusively.
I have worked mainly in gamedev, though interspersed with some fullstack work here and there.

I've also [published music](music) under various names, and [made my own video games](games). Check them out.

<!-- Here's a interactive timeline of my life: -->
<!-- `insert timeline` -->

*Last updated: 2025-08-09*

### projects

I maintain some open-source projects:

- [souvlaki](https://github.com/Sinono3/souvlaki): library for media applications to support OS media controls and "Now Playing" displays.
  Normally, the app developer would need to manually interface with each OS's API. My library abstracts all that away.
- [obsidian-helix](https://github.com/Sinono3/obsidian-helix): Obsidian plugin that enables Helix keybindings. Very simple codebase, as an underlying library is doing most of the heavy lifting.

Some notable past projects:

- [parrl](https://github.com/Sinono3/parrl): From-scratch C++ implementation of batched Cartpole env and a corresponding NN policy. Had to manually implement (1) parallel backprop, (2) parallel environments and (3) vanilla policy gradient.
  200x speed-up over Python version.
- [retroarcade](https://github.com/Sinono3/retroarcade): Libretro frontend designed for kiosk-mode Linux-based arcade machines. Handles multiple systems, cover art, controller input, resetting and auto-scanning new ROMs.
- [quiren](https://github.com/Sinono3/quiren): Minimal tool to bulk-rename filenames in the editor of your choice.
- [aldoc](https://github.com/Sinono3/aldoc): Attempt at making my own markup language that compiles to LaTeX/PDF, back when I was unsatisfied with Pandoc.
- TODO... add more here.

### posts

<ul>
{% for post in collections.posts %}
  <li><a href={{ post.url }}>{{ post.data.title }}</a>: {{ post.data.summary }} ({{ post.date | date: "%Y-%m-%d" }})</li>
{% endfor %}
</ul>

<!-- ### thoughts -->

<!-- Unfiltered notes written while trying to -->
<!-- (a) generate novel abstractions over phenomena, or -->
<!-- (b) optimize for my happiness. -->
<!-- Not directly intended to be read by an audience, but published anyway, -->
<!-- just in case someone might find it interesting or relatable. -->

<!-- <ul> -->
<!-- {% for post in collections.thoughts %} -->
  <!-- <li><a href={{ post.url }}>{{ post.data.title }}</a></li> -->
<!-- {% endfor %} -->
<!-- </ul> -->
