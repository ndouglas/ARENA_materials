---
layout: post
title:  "Triangles (0.1)"
date:   2026-11-14 18:24:18 -0500
categories: arena
---
This is postdated a bit.

I was extremely pleased that I was able to complete my implementation of `triangle_ray_intersects()` in about five minutes, just looking at the linear system - benefits of the practice I've been doing with the relevant APIs. Things appear to be working as I hoped they would, which is a very new development.

The rest - eh. It was a slog, not gonna lie. I had made myself a promise not to copy and paste any of my own code, and so I had to rederive some stuff in `raytrace_triangle()` and `raytrace_mesh()`.

More substantially, I had some kind of lapse of judgment while working on `raytrace_triangle()`. I knew what I needed to do to effectively batch the matrix operations for the `t.linalg.solve()` call, but not what I needed to do to get there. I did, basically, `einops.repeat(triangle, "pts dims -> NR pts dims", NR=NR)`, which... was not what I wanted to do.

It does make sense - a triangle for each ray - but that's not how the linear algebra addresses them. That would need something like `triangles[ray_id][A][xyz]`. The correct way is `einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)` - which gives you e.g. `triangles[A][ray_id][xyz]`. Since we're doing other things like `O` and `D` indexed the same way, it all makes sense... but this completely escaped me at the time 😩

At the end, I was left with this:

![pikachu](/assets/images/2026-11-14-01-pikachu.png)

The reward I had anticipated! 🙂
