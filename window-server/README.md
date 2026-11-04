# The window server's Metal

What it takes for CoreAnimation's window server to draw through Metal on the phone, with its history, out of
[hakios-xnu-kernel](https://github.com/oosexgang/hakios-xnu-kernel), which builds from its own copies of these files:

- `userland/src/HakiosMetalCA.m`: the plug-in that serves the window server, over this repository's `include/` and `src/`
- `drivers/gpu/`, `xnu/files/iokit/Kernel/MiniIOMobileFramebuffer.cpp`: the GPU's kernel driver, and the frames on the panel
- `tools/`: Metal on Vulkan's build (Turnip, UniXNU's AdrenoMetal, metal2vulkan), and how a run is measured
- `docs/`: the plan, what was measured

Licenses: the kernel's files here (`drivers/`, `xnu/`) are APSL 2.0, derived from XNU, not this repository's MIT;
`tools/m2v/metal2vulkan-hakios.patch` changes code under `GPL-2.0-or-later OR LGPL-3.0-or-later` and is under those terms.
The plug-in's origin is marked "to verify" in hakios-xnu-kernel's `THIRD_PARTY_LICENSES.md`.
